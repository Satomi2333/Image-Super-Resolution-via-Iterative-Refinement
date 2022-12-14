import sqlite3
import cleanfid.fid
import lpips
import numpy as np
import torch
import hashlib
from collections import OrderedDict
from torchvision.models import Inception3


class Metric:
    def __init__(self, use_db=False,
                 db_cursor: sqlite3.Cursor = None,
                 md5_cache_capacity=30000):
        self.md5_cache = OrderedDict()
        self.md5_cache_capacity = md5_cache_capacity
        self.value1 = []
        self.value12 = []
        self.value13 = []
        if use_db:
            assert db_cursor is not None
            self.db_cursor = db_cursor
        else:
            self.db_cursor = None

    def _calc_single(self, input1) -> (list, tuple):
        raise NotImplementedError()

    def _calc_double(self, input1, input2) -> (list, tuple):
        raise NotImplementedError()

    def _calc_single_and_cache(self, input1):
        cache = self._load(input1)
        if cache is not None:
            return cache
        else:
            value = self._calc_single(input1)
            self._save(input1, value)
            return value

    def final_output(self, how=np.mean, **kwargs):
        out = []
        if len(self.value1) != 0:
            value1 = how(np.array(self.value1), **kwargs)
            out.append(value1)
        if len(self.value12) != 0:
            value12 = how(np.array(self.value12), **kwargs)
            out.append(value12)
        if len(self.value13) != 0:
            value13 = how(np.array(self.value13), **kwargs)
            out.append(value13)
        return out

    def calculate(self, input1, input2=None, input3=None):
        if input2 is not None:
            if input3 is not None:
                v12, v13 = self._calc_double(input1, input2), self._calc_double(input1, input3)
                self.value12.extend(v12)
                self.value13.extend(v13)
                return v12, v13
            else:
                v12 = self._calc_double(input1, input2)
                self.value12.extend(v12)
                return v12
        else:
            v1 = self._calc_single_and_cache(input1)
            self.value1.extend(v1)
            return v1

    def __call__(self, *args, **kwargs):
        return self.calculate(*args, **kwargs)

    def _save(self, input, value):
        md5 = self._md5(input)
        self.md5_cache[md5] = value
        if len(self.md5_cache) > self.md5_cache_capacity:
            self.md5_cache.popitem()  # remove the last one (not used to access
        if self.db_cursor is not None:
            # todo: save to the database
            self.db_cursor.execute("insert into metric values ()")

    def _load(self, input):
        md5 = self._md5(input)
        value = self.md5_cache.get(md5)
        self.md5_cache.move_to_end(md5, last=False)  # move this value to the first (accessed
        if value:
            return value
        else:
            # todo:load from database
            return None

    def _md5(self, input):
        md5 = hashlib.md5(input).hexdigest()
        return md5


class PSNR(Metric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _calc_double(self,
                     input1: torch.Tensor,
                     input2: torch.Tensor) -> list[float]:
        # convert(“rgb") or not are ok
        # input [0,1]
        input1 = (input1 * 255.0).round()
        input2 = (input2 * 255.0).round()
        if len(input1.shape) == 4:
            mse = torch.mean((input1 - input2) ** 2, (1, 2, 3))
            values = []
            for m in mse:
                if m == 0:
                    values.append(float('inf'))
                else:
                    values.append((20 * torch.log10(255.0 / torch.sqrt(m))).item())
            return values


class SSIM(Metric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _calc_double(self, input1, input2) -> list[float]:
        if len(input1.shape) == 4:
            ssims = []
            for img1, img2 in zip(input1, input2):
                value = core.metrics.calculate_ssim(core.metrics.tensor2img(input1 * 2 - 1),
                                                    core.metrics.tensor2img(input2 * 2 - 1))
                ssims.append(value)
        return ssims


class LPIPS(Metric):
    def __init__(self, net="alex", gpu=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert net in ("vgg", "vgg16", "alex", "squeeze"), f"unknown type of lpips network: {net}"
        self.lpips_fn = lpips.LPIPS(net=net)
        self.use_gpu = gpu
        if self.use_gpu:
            self.lpips_fn.cuda()

    def _calc_double(self, input1, input2):
        # lpips.im2tensor(lpips.load_image(img1)) has a slight error with here (about e-8 at some pixel
        if len(input1.shape) == 4:
            lpips_list = []
            # input1 = input1 * 2 - 1  # to [-1,1] use lpips_fn.forward(,, normalize=True) instead
            # input2 = input2 * 2 - 1
            batch = input1.shape[0]
            if self.use_gpu:
                values = self.lpips_fn.forward(input1.cuda(), input2.cuda(), normalize=True).squeeze().detach().cpu()
            else:
                values = self.lpips_fn.forward(input1, input2, normalize=True).squeeze().detach().cpu()
            if batch == 1:  # values: {Tensor:()}
                lpips_list.append(values.item())
            else:  # values: {Tensor:(batch)}
                for v in values:
                    lpips_list.append(v.item())
        return lpips_list


class FID(Metric):
    def __init__(self, model_name="inception_v3", gpu=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        import cleanfid
        import torchvision.transforms.functional as F
        # https://github.com/GaParmar/clean-fid
        self.device = torch.device('cuda') if gpu else torch.device('cpu')
        self.feat_model = cleanfid.features.build_feature_extractor("clean", device=self.device)
        self.final_result = None

    def _calc_double(self, input1, input2):
        pass

    def _calc_single(self, input1):

        pass

    def final_output(self):
        mu1 = np.mean(np_feats1, axis=0)
        sigma1 = np.cov(np_feats1, rowvar=False)
        mu2 = np.mean(np_feats2, axis=0)
        sigma2 = np.cov(np_feats2, rowvar=False)
        cleanfid.fid.frechet_distance(mu1, sigma1, mu2, sigma2)


class Metrics:
    def __init__(self,
                 database: str,
                 metrics: Metric[list, tuple]):
        self.metrics = metrics
        self._init_db(database)

    def _init_db(self, database):
        self.db = sqlite3.connect(database)
        self.db_cursor = self.db.cursor()

    def calculate(self, input1, input2=None, input3=None):
        results = []
        for m in self.metrics:
            result = m(input1, input2, input3)
            results.append(result)
        return results

