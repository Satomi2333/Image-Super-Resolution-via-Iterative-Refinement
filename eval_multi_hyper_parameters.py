from collections import OrderedDict

import torch
from natsort import natsorted
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.utils import make_product
from core.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter
import os
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/derain_sr3_deblur_16_128_val.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['val'], help='val(generation)', default='val')
    parser.add_argument('-d', '--dir_of_ckpt', type=str, required=True,
                        help='path of ckpt, like experiments/sr_ffhq_../checkpoint/I1_E1')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '--debug', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_eval', action='store_true')

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

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

    to_search1 = [10, 20, 30, 50, 100, 200, 300, 500]
    to_search2 = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    def parameter_path1(opt, value): opt['model']['beta_schedule']['val']['n_timestep'] = value
    # def parameter_path1(opt, value): opt['model']['diffusion']['ddim_timesteps'] = value
    def parameter_path2(opt, value): opt['model']['beta_schedule']['val']['linear_end'] = value
    gird_search_list = [(parameter_path1, to_search1), (parameter_path2, to_search2)]
    product = make_product(gird_search_list)

    for item in product:
        # item is a list of hyper-parameters like [(func1, values1[0]), (func2, values2[0]), ..]
        for (func, value) in item:
            func(opt, value)
        # (Re)Initialize WandbLogger
        if opt['enable_wandb']:
            wandb_logger = WandbLogger(opt, reinit=True)
            wandb.define_metric("validation/*")
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

        diffusion.set_new_noise_schedule(
            opt['model']['beta_schedule']['train'], schedule_phase='train') # match the size of the scheduler,
        # or RuntimeError: Error(s) in loading state_dict for GaussianDiffusion:size mismatch for betas...
        diffusion.opt['path']['resume_state'] = args.dir_of_ckpt
        diffusion.load_network()
        diffusion.set_new_noise_schedule(
            opt['model']['beta_schedule']['val'], schedule_phase='val')

        avg_psnr = 0.0
        avg_ssim = 0.0
        idx = 0
        result_path = '{}/{}'.format(opt['path']['results'], "_".join([str(x[1]) for x in item]))
        os.makedirs(result_path, exist_ok=True)
        for _,  val_data in enumerate(val_loader):
            idx += 1
            diffusion.feed_data(val_data)
            diffusion.test(continous=False, use_ddpm_when_ddim_failed=False, threshold_psnr=20)
            visuals = diffusion.get_current_visuals()
            sr_img = Metrics.tensor2img(visuals['SR'])  # uint8
            hr_img = Metrics.tensor2img(visuals['HR'])  # uint8
            lr_img = Metrics.tensor2img(visuals['LR'])  # uint8
            fake_img = Metrics.tensor2img(visuals['INF'])  # uint8
            if visuals['Res'] is not None: res_img = Metrics.tensor2img(visuals['Res'])

            Metrics.save_img(
                sr_img, '{}/{}_sr.png'.format(result_path, idx))
            Metrics.save_img(
                hr_img, '{}/{}_hr.png'.format(result_path, idx))
            Metrics.save_img(
                lr_img, '{}/{}_lr.png'.format(result_path, idx))
            # Metrics.save_img(
            #     fake_img, '{}/{}_{}_inf.png'.format(result_path, current_step, idx))
            if visuals['Res'] is not None:
                Metrics.save_img(
                    res_img, '{}/{}_res.png'.format(result_path, idx))

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
        logger_val.info('{} psnr: {:.4e}, ssim：{:.4e}'.format(
            "_".join([str(x[1]) for x in item]), avg_psnr, avg_ssim))

        if wandb_logger:
            if opt['log_eval']:
                wandb_logger.log_eval_table()
            wandb_logger.log_metrics({
                'validation/val_psnr': avg_psnr,
                'validation/val_ssim': avg_ssim,
            })
            wandb_logger.finish()
