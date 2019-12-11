import argparse
from os import listdir
from os.path import join
import torch
from PIL import Image
from torch.autograd import Variable

from model import Generator
from torchvision.transforms import ToTensor, ToPILImage, Resize, Compose
from data_utils import is_image_file

parser = argparse.ArgumentParser(description='Test Single Image')
parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
parser.add_argument('--model_name', default='netG_epoch_4_99_128.pth', type=str, help='generator model epoch name')


with torch.no_grad():
    opt = parser.parse_args()
    UPSCALE_FACTOR = opt.upscale_factor
    MODEL_NAME = opt.model_name

    dataset_dir = 'data/test_image'
    image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
    dataset_len = len(image_filenames)

    model = Generator(UPSCALE_FACTOR).eval()
    model.cuda()
    model.load_state_dict(torch.load('epochs/' + MODEL_NAME))

    for index in range(dataset_len):
        real_image = Image.open(image_filenames[index])

        image = Variable(ToTensor()(real_image)).unsqueeze(0)
        image = image.cuda()

        out = model(image)
        out_img = ToPILImage()(out[0].data.cpu())

        #out_img = Resize((image.shape[2], image.shape[3]), interpolation=Image.BICUBIC)(out_img)
        out_img.save('result_image/out_srf_' + str(UPSCALE_FACTOR) + '_' + image_filenames[index].split('/')[2])

        out_img_bicubic = Resize((image.shape[2]*UPSCALE_FACTOR, image.shape[3]*UPSCALE_FACTOR), interpolation=Image.BICUBIC)(real_image)
        out_img_bicubic.save('result_image/out_srf_BICUBIC_' + str(UPSCALE_FACTOR) + '_' + image_filenames[index].split('/')[2])
        print(image_filenames[index].split('/')[2], "...done")



