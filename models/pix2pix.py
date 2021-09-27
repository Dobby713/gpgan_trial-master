import os
import numpy as np
import torch
from models.submodels import UnetGenerator, get_norm_layer, NLayerDiscriminator, Classifier, VGG16MultiOut
from models.losses import GANLoss
import pandas as pd
from util.image_pool import ImagePool
from util.utils import print_network
from torch.autograd import Variable
from torchvision import transforms
from collections import OrderedDict


class Pix2PixClassifierModel:
    def __init__(self, opt):
        self.opt = opt
        self.name = 'p2p'
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.gpu_ids = opt.gpu_ids
        self.Tensor = torch.Tensor

        self.input = None
        self.image_paths = None
        self.real_A = None
        self.fake_B = None
        self.real_B = None
        self.preprocess_fake_B = None
        self.perceptual_fake_B_out = None
        self.gender_fake_B_out = None
        self.preprocess_real_B = None
        self.perceptual_real_B_out = None
        self.gender_real_B_out = None

        self.input_A = self.Tensor(opt.batchSize, opt.input_nc,
                                   opt.fineSize, opt.fineSize)
        self.input_B = self.Tensor(opt.batchSize, opt.output_nc,
                                   opt.fineSize, opt.fineSize)
        norm_layer = get_norm_layer(opt.norm)
        self.netG = UnetGenerator(opt.input_nc, opt.output_nc, 6, opt.ngf, norm_layer=norm_layer,
                                  use_dropout=not opt.no_dropout, gpu_ids=self.gpu_ids)

        use_sigmoid = opt.no_lsgan
        self.netD = NLayerDiscriminator(opt.input_nc * 2, opt.ndf, n_layers=opt.n_layers_D, norm_layer=norm_layer,
                                        use_sigmoid=use_sigmoid, gpu_ids=self.gpu_ids)

        self.netC = Classifier()
        self.vgg_model = VGG16MultiOut()
        self.freader = pd.read_csv(os.path.join(opt.dataroot, 'fine_grained_attribute.txt'), header=0, sep=' ')
        self.people = self.freader['imgname'].tolist()
        self.labels = self.freader['Male']
        self.labels[self.labels == -1] = 0
        self.fake_AB_pool = ImagePool(opt.pool_size)
        self.old_lr = opt.lr
        self.old_c_lr = opt.lr * 0.01

        device = torch.device('cuda')
        self.netC.to(device)
        self.vgg_model.to(device)
        self.netG.to(device)
        self.netD.to(device)

        self.criterionGAN = GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
        self.criterionGAN.to(device)
        self.criterionL1 = torch.nn.L1Loss()
        self.criterionP = torch.nn.L1Loss()
        self.criterionC = torch.nn.BCELoss()

        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizer_C = torch.optim.Adam(self.netC.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

        print('---Networks initialized---')
        print_network(self.netG)
        print_network(self.netD)
        print_network(self.netC)
        print('--------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        # print(input_A.shape)
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        # imagename = self.image_paths[0].split('/')[-1]
        # idx = self.people.index(imagename)
        # print(self.image_paths)
        # self.gender = np.array(self.labels[idx].astype(np.float32))
        img_name_list = [self.image_paths[k].split('/')[-1] for k in range(len(self.image_paths))]
        idx_list = [self.people.index(k) for k in img_name_list]
        self.gender = np.array([[self.labels[k]] for k in idx_list]).astype(np.float32)

    def forward(self):
        self.real_A = Variable(self.input_A).cuda()
        self.fake_B = self.netG.forward(self.real_A)
        self.real_B = Variable(self.input_B).cuda()

        self.preprocess_real_B = Variable(self.input_B).cuda()
        self.preprocess_fake_B = Variable(self.fake_B).cuda()

        self.perceptual_real_B_out = self.vgg_model.forward(self.preprocess_real_B)[3]
        # self.gender_real_B_out = self.vgg_model.forward(self.preprocess_real_B)[4]

        self.perceptual_fake_B_out = self.vgg_model.forward(self.preprocess_fake_B)[3]
        self.gender_fake_B_out = self.vgg_model.forward(self.preprocess_fake_B)[4]

    def forward_C(self):
        self.real_B = Variable(self.input_B).cuda()
        self.preprocess_real_B = Variable(self.real_B).cuda()
        self.perceptual_real_B_out = self.vgg_model.forward(self.preprocess_real_B)[3]
        self.gender_real_B_out = self.vgg_model.forward(self.preprocess_real_B)[4]

    def backward_D(self):
        fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B), 1))
        self.pred_fake = self.netD.forward(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(self.pred_fake, False)

        real_AB = torch.cat((self.real_A, self.real_B), 1)
        self.pred_real = self.netD.forward(real_AB)
        self.loss_D_real = self.criterionGAN(self.pred_real, True)

        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_C(self):
        # self.attr = self.Tensor(1, 1)
        self.attr = torch.from_numpy(self.gender).float().cuda()
        self.label = Variable(self.attr)
        # print(self.label.shape)
        # print(self.gender_real_B_out.shape)
        self.classifier_real = self.netC.forward(self.gender_real_B_out)
        self.loss_C_real = self.criterionC(self.classifier_real, self.label) * 0.0001
        self.loss_C_real.backward()

    def backward_G(self):
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD.forward(fake_AB)

        self.attr = torch.from_numpy(self.gender).float().cuda()
        self.label = Variable(self.attr)

        self.classifier_fake = self.netC.forward(self.gender_fake_B_out)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_A

        self.loss_G_perceptual = self.criterionP(self.perceptual_fake_B_out, self.perceptual_real_B_out) * self.opt.lambda_P
        self.loss_C_fake = self.criterionC(self.classifier_fake, self.label) * 1
        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_perceptual + self.loss_C_fake
        self.loss_G.backward()

    def optimize_C_parameters(self):
        self.forward_C()
        self.optimizer_C.zero_grad()
        self.backward_C()
        self.optimizer_C.step()

    def optimize_parameters(self):
        self.forward()

        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def get_C_errors(self):
        # return OrderedDict([('C_gender', self.loss_C_real.data[0]), ])
        return OrderedDict([('C_gender', self.loss_C_real), ])

    def get_current_errors(self):
        return OrderedDict([('G_GAN', self.loss_G_GAN),
                    ('G_L1', self.loss_G_L1),
                            ('G_P', self.loss_G_perceptual),
                            ('G_C_fake', self.loss_C_fake),
                        ('D_real', self.loss_D_real),
                        ('D_fake', self.loss_D_fake)
                        ])

    def test(self):
        self.real_A = Variable(self.input_A, volatile=True)
        self.fake_B = self.netG.forward(self.real_A)
        self.real_B = Variable(self.input_B, volatile=True)

    def get_image_paths(self):
        return self.image_paths

    def get_current_visuals(self):
        real_A = tensor2im(self.real_A.data)
        fake_B = tensor2im(self.fake_B.data)
        real_B = tensor2im(self.real_B.data)
        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B)])

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label, gpu_ids):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            network.cuda(device=gpu_ids[0])

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        print(save_path)
        network.load_state_dict(torch.load(save_path))

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netC, 'C', label, self.gpu_ids)
        self.save_network(self.netD, 'D', label, self.gpu_ids)

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

    def update_C_rate(self):
        lrd = self.old_c_lr / self.opt.citer
        lr = self.old_c_lr - lrd
        for param_group in self.optimizer_C.param_groups:
            param_group['lr'] = lr
        print('update learning rate: %f -> %f' % (self.old_c_lr, lr))
        self.old_c_lr = lr


def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


def preprocess(image_tensor):
    tensortype = type(image_tensor.data)
    image_tensor_out = tensortype(1, 3, 224, 224)

    # print(image_tensor.data[0])
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])

    img_tensor = transform(image_tensor.data[0].cpu())
    image_tensor_out[0] = img_tensor
    image_tensor_out.cuda()
    return image_tensor_out


def preprocess_refined(image_tensor):
    # tensortype = type(image_tensor.data)
    # image_tensor_out = tensortype(opt.batchSize, 3, opt.fineSize, opt.fineSize)

    transform = transforms.Compose([
        # transforms.ToPILImage(),
        # transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])

    image_tensor_out = transform(image_tensor)
    image_tensor_out.cuda()
    return image_tensor_out


def make_unet_generator(opt):
    norm_layer = get_norm_layer(opt.norm)
    netG = UnetGenerator(opt.input_nc, opt.output_nc, 6, opt.ngf, norm_layer=norm_layer,
                         use_dropout=not opt.no_dropout, gpu_ids=opt.gpu_ids)
    return netG

