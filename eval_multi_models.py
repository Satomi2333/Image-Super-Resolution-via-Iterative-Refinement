import torch
from natsort import natsorted
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter
import os
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/derain_sr3_deblur_16_128_val.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['val'], help='val(generation)', default='val')
    parser.add_argument('-d', '--dir_of_ckpts', type=str, required=True,
                        help='path of ckpts, like experiments/sr_ffhq_..')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '--debug', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_eval', action='store_true')

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    assert os.path.isdir(args.dir_of_ckpts), f"path error, {args.dir_of_ckpts}"

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    # logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # Initialize WandbLogger
    if opt['enable_wandb']:
        import wandb
        wandb_logger = WandbLogger(opt)
        wandb.define_metric('ckpt_step')
        wandb.define_metric('ckpt_epoch')
        wandb.define_metric("validation/*", step_metric="ckpt_epoch")
        wandb.define_metric("validation/*", step_metric="ckpt_step")

    else:
        wandb_logger = None

    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'val':
            val_set = Data.create_dataset(dataset_opt, phase)
            val_loader = Data.create_dataloader(
                val_set, dataset_opt, phase)
    logger.info('Initial Dataset Finished')

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    uni_ckpt = set()
    for ckpt in os.listdir(os.path.join(args.dir_of_ckpts, 'checkpoint')):
        uni_ckpt.add(ckpt[:-8]) # without (_gen|_opt).pth
    ckpts = natsorted(list(uni_ckpt))

    for ckpt in ckpts:
        diffusion.set_new_noise_schedule(
            opt['model']['beta_schedule']['train'], schedule_phase='train') # match the size of the scheduler,
        # or RuntimeError: Error(s) in loading state_dict for GaussianDiffusion:size mismatch for betas...
        diffusion.opt['path']['resume_state'] = os.path.join(args.dir_of_ckpts, 'checkpoint', ckpt)
        diffusion.load_network()
        optimizer = torch.load('{}_opt.pth'.format(diffusion.opt['path']['resume_state']))
        diffusion.set_new_noise_schedule(
            opt['model']['beta_schedule']['val'], schedule_phase='val')

        # step and epoch
        current_step = optimizer['iter']
        current_epoch = optimizer['epoch']

        avg_psnr = 0.0
        avg_ssim = 0.0
        idx = 0
        result_path = '{}/I{}_E{}'.format(opt['path']['results'], current_step, current_epoch)
        os.makedirs(result_path, exist_ok=True)
        for _,  val_data in enumerate(val_loader):
            idx += 1
            diffusion.feed_data(val_data)
            diffusion.test(continous=True, use_ddpm_when_ddim_failed=True, threshold_psnr=20)
            visuals = diffusion.get_current_visuals()
            sr_img = Metrics.tensor2img(visuals['SR'])  # uint8
            hr_img = Metrics.tensor2img(visuals['HR'])  # uint8
            lr_img = Metrics.tensor2img(visuals['LR'])  # uint8
            fake_img = Metrics.tensor2img(visuals['INF'])  # uint8

            Metrics.save_img(
                sr_img, '{}/{}_{}_sr.png'.format(result_path, current_step, idx))
            Metrics.save_img(
                hr_img, '{}/{}_{}_hr.png'.format(result_path, current_step, idx))
            Metrics.save_img(
                lr_img, '{}/{}_{}_lr.png'.format(result_path, current_step, idx))
            # Metrics.save_img(
            #     fake_img, '{}/{}_{}_inf.png'.format(result_path, current_step, idx))

            # generation
            eval_psnr = Metrics.calculate_psnr(Metrics.tensor2img(visuals['SR'][-1]), hr_img)
            eval_ssim = Metrics.calculate_ssim(Metrics.tensor2img(visuals['SR'][-1]), hr_img)

            avg_psnr += eval_psnr
            avg_ssim += eval_ssim

            if wandb_logger and opt['log_eval']:
                wandb_logger.log_eval_data(fake_img, Metrics.tensor2img(visuals['SR'][-1]), hr_img, eval_psnr, eval_ssim)

        avg_psnr = avg_psnr / idx
        avg_ssim = avg_ssim / idx

        # log
        logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
        logger.info('# Validation # SSIM: {:.4e}'.format(avg_ssim))
        logger_val = logging.getLogger('val')  # validation logger
        logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}, ssim：{:.4e}'.format(
            current_epoch, current_step, avg_psnr, avg_ssim))

        if wandb_logger:
            if opt['log_eval']:
                wandb_logger.log_eval_table()
            wandb_logger.log_metrics({
                'validation/val_psnr': avg_psnr,
                'validation/val_ssim': avg_ssim,
                'ckpt_epoch': current_epoch,
                'ckpt_step': current_step
            })
