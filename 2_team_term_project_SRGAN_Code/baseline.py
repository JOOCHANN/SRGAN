import argparse
from math import log10
import os

import torch.utils.data
from torch.utils.data import DataLoader
from tqdm import tqdm

import pytorch_ssim
from data_utils import ValDatasetFromFolder

parser = argparse.ArgumentParser(description='Train Super Resolution Models')
parser.add_argument('--crop_size', default=128, type=int, help='training images crop size') #128
parser.add_argument('--upscale_factor', default=2, type=int, choices=[2, 4, 8],
                    help='super resolution upscale factor')
parser.add_argument('--num_epochs', default=1, type=int, help='train epoch number')

opt = parser.parse_args()

CROP_SIZE = opt.crop_size
UPSCALE_FACTOR = opt.upscale_factor
NUM_EPOCHS = opt.num_epochs

def main():
    val_set = ValDatasetFromFolder('data/DIV2K_valid_HR', upscale_factor=UPSCALE_FACTOR)
    val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=1, shuffle=False)

    results = {'psnr': [], 'ssim': []}
    torch.manual_seed(1234)
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    for epoch in range(1, NUM_EPOCHS + 1):

        val_bar = tqdm(val_loader)
        valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
        val_images = []

        for val_lr, val_hr_restore, val_hr in val_bar:
            batch_size = val_lr.size(0)  # Batch size
            valing_results['batch_sizes'] += batch_size
            sr = val_hr_restore  # LR image
            hr = val_hr  # HR real image
            sr = sr.cuda()
            hr = hr.cuda()

            batch_mse = ((sr - hr) ** 2).data.mean()  # generator가 만든 SR image와 HR real image간의 MSE값.
            valing_results['mse'] += batch_mse * batch_size
            batch_ssim = pytorch_ssim.ssim(sr, hr).item()  # generator가 만든 SR image와 HR real image간의 SSIM값.
            valing_results['ssims'] += batch_ssim * batch_size
            # valing_results['psnr'] = 10 * log10(1*1 / (valing_results['mse'] / valing_results['batch_sizes'])) #PSNR값, 1은 Max
            valing_results['psnr'] = 20 * log10(1) - 10 * log10(
                valing_results['mse'] / valing_results['batch_sizes'])  # PSNR값, 1은 Max
            valing_results['ssim'] = valing_results['ssims'] / valing_results['batch_sizes']  # ssim값 구하기
            val_bar.set_description(  # PSNR, SSIM 출력
                desc='[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f' % (
                    valing_results['psnr'], valing_results['ssim']))


if __name__ == '__main__':
    main()
