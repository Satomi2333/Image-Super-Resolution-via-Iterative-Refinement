import argparse
import core.metrics as Metrics
from PIL import Image
import numpy as np
import glob
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str,
                        default=r'experiments\derain_sr3deblur_val_221204_105815\results\I420000_E503')
    args = parser.parse_args()
    real_names = list(glob.glob('{}/*_hr.png'.format(args.path)))
    fake_names = list(glob.glob('{}/*_sr.png'.format(args.path)))
    input_names = list(glob.glob('{}/*_lr.png'.format(args.path)))

    real_names.sort()
    fake_names.sort()
    input_names.sort()

    avg_psnr = np.array([.0, .0])
    avg_ssim = np.array([.0, .0])
    avg_lpips = np.array([.0, .0])
    idx = 0
    for rname, fname, iname in zip(real_names, fake_names, input_names):
        idx += 1
        ridx = rname.rsplit("_hr", maxsplit=1)[0]
        fidx = fname.rsplit("_sr", maxsplit=1)[0]
        iidx = iname.rsplit("_lr", maxsplit=1)[0]
        assert ridx == fidx == iidx, f'Image ridx:{ridx}!=fidx:{fidx}!=iidx:{iidx}'

        # hr_img = np.array(Image.open(rname).convert("YCbCr"))
        hr_img = np.array(Image.open(rname))
        sr_img = np.array(Image.open(fname))
        lr_img = np.array(Image.open(iname))
        psnr = Metrics.calculate_mutil_img(hr_img, lr_img, sr_img, Metrics.calculate_psnr)
        # psnr = Metrics.calculate_psnr(sr_img, hr_img)
        ssim = Metrics.calculate_mutil_img(hr_img, lr_img, sr_img, Metrics.calculate_ssim)
        # ssim = Metrics.calculate_ssim(sr_img, hr_img)
        lpips = Metrics.calculate_mutil_img(rname, iname, fname, Metrics.calculate_lpips)
        # lpips = Metrics.calculate_lpips(rname, fname)
        avg_psnr += psnr
        avg_ssim += ssim
        avg_lpips += lpips
        if idx % 20 == 0:
            print('Image:{}, PSNR:{:.4f}, SSIM:{:.4f}, LPIPS:{:.4f}\nImage:{}, PSNR:{:.4f}, SSIM:{:.4f}, LPIPS:{:.4f}'
                  .format(idx, psnr[0], ssim[0], lpips[0], idx, psnr[1], ssim[1], lpips[1]))

    avg_psnr = avg_psnr / idx
    avg_ssim = avg_ssim / idx
    avg_lpips = avg_lpips / idx

    # log
    print('# Validation # PSNR: {:.4f}, {:.4f}'.format(avg_psnr[0], avg_psnr[1]))
    print('# Validation # SSIM: {:.4f}, {:.4f}'.format(avg_ssim[0], avg_ssim[1]))
    print('# Validation # LPIPS: {:.4f}, {:.4f}'.format(avg_lpips[0], avg_lpips[1]))
