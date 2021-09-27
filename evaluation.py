import torch
import os
from models.pix2pix import make_unet_generator
from options.train_options import TrainOptions
from torchvision import transforms
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt


class ProcessImage:
    def __init__(self):
        transform_list = [transforms.ToTensor(), transforms.Resize(256), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        self.transform = transforms.Compose(transform_list)

    def preprocess_image(self, path):
        img = Image.open(path).convert('RGB')
        return self.transform(img)


class ModelEval:
    def __init__(self, model_path):
        self.opt = TrainOptions().parse()
        self.model = make_unet_generator(self.opt)
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.trans = ProcessImage()

        mother_path = '/home/hsha/data/gpgan/datasets/lfw'
        self.Apath = os.path.join(mother_path, 'trainB')
        self.Bpath = os.path.join(mother_path, 'trainA')

        self.Alist = glob(os.path.join(self.Apath, '*'))

    def view_single(self, idx):
        sample_path = self.Alist[idx]
        sample_name = sample_path.split('/')[-1].split('_B.jpg')[0]
        sample_ypath = os.path.join(self.Bpath, sample_name + '_A.jpg')

        inputs = self.trans.preprocess_image(sample_path)
        ans = self.trans.preprocess_image(sample_ypath)

        y = self.model(inputs.unsqueeze(0))

        plt.imshow(transforms.functional.rotate(inputs, 90).transpose(0, 2))
        plt.show()

        plt.imshow(transforms.functional.rotate(y[0].detach(), 90).transpose(0, 2))
        plt.show()

        plt.imshow(transforms.functional.rotate(ans, 90).transpose(0, 2))
        plt.show()

    def view_list(self, input_list):
        for k in input_list:
            self.view_single(k)


if __name__ == '__main__':
    model_path = './checkpoints/experiment_name/latest_net_G.pth'
    me = ModelEval(model_path)
    me.view_list([0, 5, 10, 55])
    print(1)
