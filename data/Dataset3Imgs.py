import glob
import torch
import natsort
import torchvision
from PIL import Image


class Dataset3Imgs(torch.utils.data.Dataset):
    '''
    Can create a dataset with 3 imgs in once read (2 or 1 can also work)
    The best way to create this dataset is by parsing a glob like path, like "/*_lr.jpg"
    parsing a list or tuple is ok
    '''

    def __init__(self, imgs1,
                 imgs2=None,
                 imgs3=None,
                 transforms=[],
                 ):
        super(Dataset3Imgs, self).__init__()
        if isinstance(imgs1, str):
            imgs1_list = glob.glob(imgs1)
        elif isinstance(imgs1, (list, tuple)):
            imgs1_list = imgs1
        else:
            raise NotImplementedError(f"got unknown type of imgs1:{type(imgs1)}")
        self.imgs1_list = natsort.natsorted(imgs1_list)
        if imgs2 is not None:
            if isinstance(imgs2, str):
                imgs2_list = glob.glob(imgs2)
            elif isinstance(imgs2, (list, tuple)):
                imgs2_list = imgs2
            self.imgs2_list = natsort.natsorted(imgs2_list)
            assert len(imgs1_list) == len(imgs2_list), "imgs1 should have same length with imgs2"
        else:
            self.imgs2_list = None
        if imgs3 is not None:
            if isinstance(imgs3, str):
                imgs3_list = glob.glob(imgs3)
            elif isinstance(imgs3, (list, tuple)):
                imgs3_list = imgs3
            self.imgs3_list = natsort.natsorted(imgs3_list)
            assert len(imgs2_list) == len(imgs3_list), "imgs2 should have same length with imgs3"
        else:
            self.imgs3_list = None
        trans = [torchvision.transforms.ToTensor()]
        if isinstance(transforms, (list, tuple)):
            trans.extend(transforms)
        elif isinstance(transforms, torchvision.transforms.transforms):
            trans.append(transforms)
        else:
            raise TypeError(f"got unknown type of transforms:{type(transforms)}")
        self.transforms = torchvision.transforms.Compose(trans)

    def __len__(self):
        return len(self.imgs1_list)

    def __getitem__(self, item):
        files = self.get_file(item)
        for k, v in files.items():
            if v is None:
                continue
            img = Image.open(v)  # .convert("RGB")
            files[k] = self.transforms(img)
        return files

    def get_file(self, item):
        if self.imgs2_list is not None:
            if self.imgs3_list is not None:
                return {"img1": self.imgs1_list[item], "img2": self.imgs2_list[item], "img3": self.imgs3_list[item]}
            else:
                return {"img1": self.imgs1_list[item], "img2": self.imgs2_list[item], "img3": None}
        else:
            return {"img1": self.imgs1_list[item], "img2": None, "img3": None}

