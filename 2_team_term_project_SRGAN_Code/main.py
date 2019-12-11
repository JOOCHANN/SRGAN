import argparse
import os
from math import log10

import pandas as pd
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn

import pytorch_ssim
from data_utils import TrainDatasetFromFolder, ValDatasetFromFolder
from loss import GeneratorLoss
from model import Generator, Discriminator

parser = argparse.ArgumentParser(description='Train Super Resolution Models')
parser.add_argument('--crop_size', default=88, type=int, help='training images crop size') #88, 128, 168
parser.add_argument('--upscale_factor', default=4, type=int, choices=[2, 4, 8],
                    help='super resolution upscale factor')
parser.add_argument('--num_epochs', default=150, type=int, help='train epoch number')

opt = parser.parse_args()

CROP_SIZE = opt.crop_size
UPSCALE_FACTOR = opt.upscale_factor
NUM_EPOCHS = opt.num_epochs

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2, 3'

    train_set = TrainDatasetFromFolder('data/DIV2K_train_HR', crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR)
    val_set = ValDatasetFromFolder('data/DIV2K_valid_HR', upscale_factor=UPSCALE_FACTOR)
    train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=64, shuffle=True)
    val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=1, shuffle=False)

    netG = Generator(UPSCALE_FACTOR)
    print('# generator parameters:', sum(param.numel() for param in netG.parameters())) # Generator의 총 parameter 수
    netD = Discriminator()
    print('# discriminator parameters:', sum(param.numel() for param in netD.parameters())) # Discriminator의 총 parameter 수

    generator_criterion = GeneratorLoss()  # loss function
    netG = nn.DataParallel(netG).cuda()
    netD = nn.DataParallel(netD).cuda()
    #netG = netG.cuda()
    #netD = netD.cuda()
    generator_criterion.cuda()

    optimizerG = optim.Adam(netG.parameters())  # optimizer : adam
    optimizerD = optim.Adam(netD.parameters())  # optimizer : adam

    results = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': []}

    for epoch in range(1, NUM_EPOCHS + 1):
        # model train
        d_loss, g_loss, d_score, g_score = train(netG, netD, generator_criterion, optimizerG, optimizerD, train_loader, epoch)

        # validation data acc
        psnr, ssim = test(netG, netD, val_loader, epoch)

        # save loss\scores\psnr\ssim
        results['d_loss'].append(d_loss)
        results['g_loss'].append(g_loss)
        results['d_score'].append(d_score)
        results['g_score'].append(g_score)
        results['psnr'].append(psnr)
        results['ssim'].append(ssim)

        # save results
        if epoch % 10 == 0 and epoch != 0:
            out_path = 'statistics/'
            data_frame = pd.DataFrame(
                data={'Loss_D': results['d_loss'], 'Loss_G': results['g_loss'], 'Score_D': results['d_score'],
                      'Score_G': results['g_score'], 'PSNR': results['psnr'], 'SSIM': results['ssim']},
                index=range(1, epoch + 1))
            data_frame.to_csv(out_path + 'srf_' + str(UPSCALE_FACTOR) + '_train_results.csv', index_label='Epoch')

        print()


def train(netG, netD, generator_criterion, optimizerG, optimizerD, train_loader, epoch):
    train_bar = tqdm(train_loader)  # bar 형태로 표현
    running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}
    netG.train()
    netD.train()
    for data, target in train_bar:
        batch_size = data.size(0)  # Batch size
        running_results['batch_sizes'] += batch_size #모든 Batch_size의 합

        ############################
        # (1) Update D network: maximize D(x)-1-D(G(z))
        ###########################
        real_img = Variable(target)  # HR real image
        real_img = real_img.cuda()
        z = Variable(data)  # LR image size : (crop_size//upscale_factor, crop_size//upscale_factor)
        z = z.cuda()

        fake_img = netG(z)  # generator가 만든 SR image (crop_size//upscale_factor, crop_size//upscale_factor) -> (crop_size, crop_size)
        netD.zero_grad()  # netD의 gradient를 0으로 초기화
        real_out = netD(real_img).mean()  # Discriminator가 판별한 HR real image가 진짜 그림인지에 대한 output의 Batch별 평균값 0~1
        fake_out = netD(fake_img).mean()  # Discriminator가 판별한 generator가 만든SR image가 진짜 그림인지에 대한 output의 Batch별 평균값 0~1
        d_loss = 1 - real_out + fake_out  # real_out은 1이 되도록 fake_out은 0이 되도록 loss를 측정
        # After loss.backward you cannot do another loss.backward unless retain_variables is true.
        d_loss.backward(retain_graph=True)  # d_loss backpropagation, retain_graph=True 이유는 d_loss이후에 g_loss를 구해주어야 하기 때문에 retain_graph=True로 둬야함

        optimizerD.step()  # update netD weight

        ############################
        # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
        ###########################
        netG.zero_grad()  # netG의 gradient를 0으로 초기화
        g_loss = generator_criterion(fake_out, fake_img, real_img)  # g_loss 측정
        g_loss.backward()  # g_loss backpropagation
        fake_img = netG(z)  # generator가 만든 SR image
        fake_out = netD(fake_img).mean()  # Discriminator가 판별한 generator가 만든SR image가 진짜 그림인지에 대한 output의 Batch별 평균값 0~1

        optimizerG.step()  # update netG weight

        # loss for current batch before optimization
        running_results['g_loss'] += g_loss.item() * batch_size
        running_results['d_loss'] += d_loss.item() * batch_size
        running_results['d_score'] += real_out.item() * batch_size
        running_results['g_score'] += fake_out.item() * batch_size

        train_bar.set_description(desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f' % (
            epoch, NUM_EPOCHS, running_results['d_loss'] / running_results['batch_sizes'],
            running_results['g_loss'] / running_results['batch_sizes'],
            running_results['d_score'] / running_results['batch_sizes'],
            running_results['g_score'] / running_results['batch_sizes']))

    d_loss = running_results['d_loss'] / running_results['batch_sizes']
    g_loss = running_results['g_loss'] / running_results['batch_sizes']
    d_score = running_results['d_score'] / running_results['batch_sizes']
    g_score = running_results['g_score'] / running_results['batch_sizes']

    return d_loss, g_loss, d_score, g_score


def test(netG, netD, val_loader, epoch):
    netG.eval()
    # validation data 측정
    with torch.no_grad():
        val_bar = tqdm(val_loader)
        valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
        val_images = []
        for val_lr, val_hr_restore, val_hr in val_bar:
            batch_size = val_lr.size(0)  # Batch size
            valing_results['batch_sizes'] += batch_size
            lr = val_lr  # LR image
            hr = val_hr  # HR real image
            if torch.cuda.is_available():
                lr = lr.cuda()
                hr = hr.cuda()
            sr = netG(lr)  # generator가 만든 SR image

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

    # save model parameters
    torch.save(netG.state_dict(),
               'epochs/netG_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))  # Generator의 parameters 저장
    torch.save(netD.state_dict(),
               'epochs/netD_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))  # Discriminator의 parameters 저장

    psnr = valing_results['psnr']
    ssim = valing_results['ssim']
    return psnr, ssim

if __name__ == '__main__':
    main()
