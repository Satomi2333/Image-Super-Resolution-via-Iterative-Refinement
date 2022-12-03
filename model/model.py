import logging
from collections import OrderedDict
import copy

import torch
import torch.nn as nn
import os
import model.networks as networks
from .base_model import BaseModel
logger = logging.getLogger('base')


class EMA():
    def __init__(self, beta=0.9999):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class DDPM(BaseModel):
    def __init__(self, opt):
        super(DDPM, self).__init__(opt)
        # define network and load pretrained models
        self.netG = self.set_device(networks.define_G(opt))
        self.schedule_phase = None
        if opt['train']['ema_scheduler'] is not None:
            self.ema_scheduler = opt['train']['ema_scheduler']
            self.netG_EMA = copy.deepcopy(self.netG)
            self.netG_EMA = self.set_device(self.netG_EMA)
            self.EMA = EMA(beta=self.ema_scheduler['ema_decay'])
        else:
            self.ema_scheduler = None

        # set loss and load resume state
        self.set_loss()
        self.set_new_noise_schedule(
            opt['model']['beta_schedule']['train'], schedule_phase='train')
        if self.opt['phase'] == 'train':
            self.netG.train()
            # find the parameters to optimize
            if opt['model']['finetune_norm']:
                optim_params = []
                for k, v in self.netG.named_parameters():
                    v.requires_grad = False
                    if k.find('transformer') >= 0:
                        v.requires_grad = True
                        v.data.zero_()
                        optim_params.append(v)
                        logger.info(
                            'Params [{:s}] initialized to 0 and will optimize.'.format(k))
            else:
                optim_params = list(self.netG.parameters())

            self.optG = torch.optim.Adam(
                optim_params, lr=opt['train']["optimizer"]["lr"])
            self.log_dict = OrderedDict()
        self.load_network()
        self.print_network()

    def feed_data(self, data):
        self.data = self.set_device(data)

    def optimize_parameters(self, steps=0):
        self.optG.zero_grad()
        l_pix = self.netG(self.data)
        # need to average in multi-gpu
        b, c, h, w = self.data['HR'].shape
        l_pix = l_pix.sum()/int(b*c*h*w)
        l_pix.backward()
        self.optG.step()
        if self.ema_scheduler is not None:
            if steps > self.ema_scheduler['step_start_ema'] and steps % self.ema_scheduler['update_ema_every'] == 0:
                self.EMA.update_model_average(self.netG_EMA, self.netG)
        # set log
        self.log_dict['l_pix'] = l_pix.item()

    def test(self, continous=False, use_ddpm_when_ddim_failed=False, threshold_psnr=15):
        if self.ema_scheduler: self.netG_EMA.eval()
        self.netG.eval()
        with torch.no_grad():
            if self.ema_scheduler:
                if isinstance(self.netG_EMA, nn.DataParallel):
                    self.SR = self.netG_EMA.module.super_resolution(
                        self.data['SR'], continous)
                else:
                    self.SR = self.netG_EMA.super_resolution(
                        self.data['SR'], continous)
            else:
                if isinstance(self.netG, nn.DataParallel):
                    self.SR = self.netG.module.super_resolution(
                        self.data['SR'], continous)
                else:
                    self.SR = self.netG.super_resolution(
                        self.data['SR'], continous)
        if use_ddpm_when_ddim_failed:
            visuals = self.get_current_visuals()
            hr_img = Metrics.tensor2img(visuals['HR'])  # uint8
            psnr = Metrics.calculate_psnr(Metrics.tensor2img(visuals['SR'][-1]), hr_img)
            if psnr < threshold_psnr:
                imshow(hr_img)
                print("ddim may failed, psnr: ", psnr)
                if isinstance(self.netG, nn.DataParallel):
                    self.netG.module.use_ddim = False
                    self.SR = self.netG.module.super_resolution(
                        self.data['SR'], continous)
                    self.netG.module.use_ddim = True
                else:
                    self.netG.use_ddim = False
                    self.SR = self.netG.super_resolution(
                        self.data['SR'], continous)
                    self.netG.use_ddim = True
        self.netG.train()
        if self.ema_scheduler: self.netG_EMA.train() # perhaps not necessary

    def sample(self, batch_size=1, continous=False):
        self.netG.eval()
        if self.ema_scheduler: self.netG_EMA.eval()
        with torch.no_grad():
            if self.ema_scheduler:
                if isinstance(self.netG_EMA, nn.DataParallel):
                    self.SR = self.netG_EMA.module.sample(batch_size, continous)
                else:
                    self.SR = self.netG_EMA.sample(batch_size, continous)
            else:
                if isinstance(self.netG, nn.DataParallel):
                    self.SR = self.netG.module.sample(batch_size, continous)
                else:
                    self.SR = self.netG.sample(batch_size, continous)
        self.netG.train()
        if self.ema_scheduler: self.netG_EMA.train()

    def set_loss(self):
        if isinstance(self.netG, nn.DataParallel):
            self.netG.module.set_loss(self.device)
        else:
            self.netG.set_loss(self.device)

    def set_new_noise_schedule(self, schedule_opt, schedule_phase='train'):
        if self.schedule_phase is None or self.schedule_phase != schedule_phase:
            self.schedule_phase = schedule_phase
            if isinstance(self.netG, nn.DataParallel):
                self.netG.module.set_new_noise_schedule(
                    schedule_opt, self.device)
            else:
                self.netG.set_new_noise_schedule(schedule_opt, self.device)

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_LR=True, sample=False):
        out_dict = OrderedDict()
        if sample:
            out_dict['SAM'] = self.SR.detach().float().cpu()
        else:
            out_dict['SR'] = self.SR.detach().float().cpu()
            out_dict['INF'] = self.data['SR'].detach().float().cpu()
            out_dict['HR'] = self.data['HR'].detach().float().cpu()
            if need_LR and 'LR' in self.data:
                out_dict['LR'] = self.data['LR'].detach().float().cpu()
            else:
                out_dict['LR'] = out_dict['INF']
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)

        logger.info(
            'Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)

    def save_network(self, epoch, iter_step):
        gen_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_gen.pth'.format(iter_step, epoch))
        opt_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_opt.pth'.format(iter_step, epoch))
        # gen
        network = self.netG
        if isinstance(self.netG, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, gen_path)
        # opt
        opt_state = {'epoch': epoch, 'iter': iter_step,
                     'scheduler': None, 'optimizer': None}
        opt_state['optimizer'] = self.optG.state_dict()
        torch.save(opt_state, opt_path)
        # ema
        if self.ema_scheduler:
            ema_path = os.path.join(
                self.opt['path']['checkpoint'], 'I{}_E{}_ema.pth'.format(iter_step, epoch))
            network_ema = self.netG_EMA
            if isinstance(self.netG_EMA, nn.DataParallel):
                network_ema = network_ema.module
            state_dict = network_ema.state_dict()
            for key, param in state_dict.items():
                state_dict[key] = param.cpu()
            torch.save(state_dict, ema_path)

        logger.info(
            'Saved model in [{:s}] ...'.format(gen_path))

    def load_network(self):
        load_path = self.opt['path']['resume_state']
        if load_path is not None:
            logger.info(
                'Loading pretrained model for G [{:s}] ...'.format(load_path))
            gen_path = '{}_gen.pth'.format(load_path)
            opt_path = '{}_opt.pth'.format(load_path)
            # gen
            network = self.netG
            if isinstance(self.netG, nn.DataParallel):
                network = network.module
            network.load_state_dict(torch.load(
                gen_path), strict=(not self.opt['model']['finetune_norm']))
            # network.load_state_dict(torch.load(
            #     gen_path), strict=False)
            if self.opt['phase'] == 'train':
                # optimizer
                opt = torch.load(opt_path)
                self.optG.load_state_dict(opt['optimizer'])
                self.begin_step = opt['iter']
                self.begin_epoch = opt['epoch']
            # ema
            if self.ema_scheduler:
                ema_path = '{}_ema.pth'.format(load_path)
                if not os.path.exists(ema_path):
                    logger.info(
                        '{}_ema.pth not found, will disable ema'.format(load_path))
                    self.ema_scheduler = None
                    return
                network = self.netG_EMA
                if isinstance(self.netG_EMA, nn.DataParallel):
                    network = network.module
                network.load_state_dict(torch.load(
                    ema_path), strict=(not self.opt['model']['finetune_norm']))
