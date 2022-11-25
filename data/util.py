import os
import torch
import torchvision
import random
import numpy as np
from torchvision.transforms import functional as F

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG',
                  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def get_paths_from_images(path):
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return sorted(images)


def augment(img_list, hflip=True, rot=True, split='val'):
    # horizontal flip OR rotate
    hflip = hflip and (split == 'train' and random.random() < 0.5)
    vflip = rot and (split == 'train' and random.random() < 0.5)
    rot90 = rot and (split == 'train' and random.random() < 0.5)

    def _augment(img):
        if hflip:
            img = img[:, ::-1, :]
        if vflip:
            img = img[::-1, :, :]
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    return [_augment(img) for img in img_list]


def transform2numpy(img):
    img = np.array(img)
    img = img.astype(np.float32) / 255.
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    # some images have 4 channels
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img


def transform2tensor(img, min_max=(0, 1)):
    # HWC to CHW
    img = torch.from_numpy(np.ascontiguousarray(
        np.transpose(img, (2, 0, 1)))).float()
    # to range min_max
    img = img*(min_max[1] - min_max[0]) + min_max[0]
    return img


# implementation by numpy and torch
# def transform_augment(img_list, split='val', min_max=(0, 1)):
#     imgs = [transform2numpy(img) for img in img_list]
#     imgs = augment(imgs, split=split)
#     ret_img = [transform2tensor(img, min_max) for img in imgs]
#     return ret_img


# implementation by torchvision, detail in https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement/issues/14
totensor = torchvision.transforms.ToTensor()
hflip = torchvision.transforms.RandomHorizontalFlip()
def transform_augment(img_list, split='val', min_max=(0, 1), image_size=128):
    imgs = [totensor(img) for img in img_list]
    # if len(imgs) == 2 and any([i.shape()[1] != i.shape()[2] != image_size for i in imgs]):
    # in the file torchvision\transforms\transforms.py class RandomCrop, crop size checker is existed
    if len(imgs) == 2:
        # do crop when got input [img_SR, img_HR]
        crop = RandomCropMulti(image_size)
        imgs = crop(imgs)
    if split == 'train':
        imgs = torch.stack(imgs, 0)
        imgs = hflip(imgs)
        imgs = torch.unbind(imgs, dim=0)
    ret_img = [img * (min_max[1] - min_max[0]) + min_max[0] for img in imgs]
    return ret_img

class RandomCropMulti(torchvision.transforms.RandomCrop):
    def forward(self, img_list):
        """
        Args:
            img_list (List of img): Images to be cropped.
            img (PIL Image or Tensor): Image to be cropped.
            require: img1.size == img2.size == ... (imgn from img_list)

        Returns:
            img_list (List of img): Cropped images.
        """
        if self.padding is not None:
            img_list = [F.pad(img, self.padding, self.fill, self.padding_mode) for img in img_list]

        _, height, width = F.get_dimensions(img_list[0])
        # pad the width if needed
        if self.pad_if_needed and width < self.size[1]:
            padding = [self.size[1] - width, 0]
            img_list = [F.pad(img, padding, self.fill, self.padding_mode) for img in img_list]
        # pad the height if needed
        if self.pad_if_needed and height < self.size[0]:
            padding = [0, self.size[0] - height]
            img_list = [F.pad(img, padding, self.fill, self.padding_mode) for img in img_list]

        i, j, h, w = self.get_params(img_list[0], self.size)

        return [F.crop(img, i, j, h, w) for img in img_list]