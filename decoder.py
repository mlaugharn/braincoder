# https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
import config

import torch.nn as nn
import einops

class ResidualBlock(nn.Module):
    def __init__(self, module):
        super(ResidualBlock, self).__init__()
        self.module = module
    
    def forward(self, inputs):
        return self.module(inputs) + inputs

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x):
        return self.layer(x)



class ResizeConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=False):
        super(ResizeConvolution, self).__init__()
        self.layers = nn.Sequential(
            nn.Upsample(scale_factor = 2, mode='bilinear'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.GELU(),
            nn.BatchNorm2d(out_channels)
        )
    def forward(self, x):
        return self.layers(x)


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)

def make_decoder(NUM_VOXELS):
    ngf = 64 # num generator features? idk
    feat_block_size = 8
    h = 4 # sqrt(128 / 8)
    w = 4
    feat_channels = ngf * 8
    return nn.Sequential(
    # nn.Linear(NUM_VOXELS, NUM_VOXELS//16),
    # nn.BatchNorm1d(NUM_VOXELS//16),
    # nn.ReLU(),
        # nn.Linear(NUM_VOXELS, NUM_VOXELS // 4),
        # nn.BatchNorm1d(NUM_VOXELS // 4),
        # nn.Linear(NUM_VOXELS // 4, h * w * ngf),
        # einops.layers.torch.Rearrange('b (c h w) -> b c h w', h=h, w=w),
        # ResidualBlock(DecoderBlock(ngf, ngf)),
            # nn.Linear(NUM_VOXELS, NUM_VOXELS // 4),
            # nn.ReLU(),
        nn.Linear(NUM_VOXELS, 512 * h * w),
        # einops.layers.torch.Rearrange('b N -> b N 1 1'),
        # DecoderBlock(NUM_VOXELS, ngf),
        # nn.Upsample(scale_factor=2),
        # DecoderBlock(ngf, ngf // 2),
        # nn.Upsample(scale_factor=2),
        # DecoderBlock(ngf // 2, ngf // 4),
        # nn.Upsample(scale_factor=2),
        # DecoderBlock(ngf // 4, ngf // 8),
        # nn.Upsample(scale_factor=2),
        # DecoderBlock(ngf // 8, ngf // 16),
        # nn.Upsample(scale_factor=2),
        # DecoderBlock(ngf // 16, ngf // 32),
        # nn.Upsample(scale_factor=2),
        # DecoderBlock(ngf // 32, ngf // 32),
        # # nn.ConvTranspose2d(feat_channels, ngf, kernel_size=3, stride=1, padding=0, bias=False),
        # # nn.ReLU(True),
        # # # nn.Upsample(2),
        # # # nn.BatchNorm2d(ngf),
        # # nn.ConvTranspose2d(feat_channels, ngf, kernel_size=3, stride=1, padding=0, bias=False),
        # # nn.ReLU(True),
        # # # nn.Upsample(2),
        # # # nn.BatchNorm2d(ngf),
        # # nn.ConvTranspose2d(feat_channels, ngf, kernel_size=3, stride=1, padding=0, bias=False),
        # # nn.ReLU(True),
        # # # nn.Upsample(2),
        # # # nn.BatchNorm2d(ngf),
        # nn.ConvTranspose2d(ngf // 32, 3, kernel_size=3, stride=1, padding=1, bias=False),
        # nn.Upsample(size=(config.img_size, config.img_size)),
        # nn.Sigmoid()
        Reshape(-1, 512, h, w),
        # nn.ConvTranspose2d(48, 24, kernel_size=4, stride=2, padding=0, bias=False),
        # nn.BatchNorm2d(ngf),
        # nn.LeakyReLU(True),
        # nn.ConvTranspose2d(ngf, ngf, kernel_size=4, stride=2, padding=1, bias=False),
        # nn.BatchNorm2d(ngf),
        # nn.LeakyReLU(True),
        # nn.ConvTranspose2d(ngf, ngf, kernel_size=4, stride=2, padding=1, bias=False),
        # nn.BatchNorm2d(ngf),
        # nn.LeakyReLU(True),
        
        # nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
        # nn.BatchNorm2d(ngf * 2),
        # nn.ReLU(True),
        # nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=3, stride=2, padding=1, bias=False),
        # nn.BatchNorm2d(ngf),
        # nn.ConvTranspose2d(ngf, 3, kernel_size=3, stride=2, padding=1, bias=False),

        # nn.ConvTranspose2d(NUM_VOXELS, ngf * 32, kernel_size=4, stride=2, padding=0, bias=False),
        # ResizeConvolution(ngf * 32, ngf * 8),
        # nn.BatchNorm2d(ngf * 8),
        # nn.ReLU(True),
        # ResizeConvolution(ngf * 8, ngf * 4),
        # nn.BatchNorm2d(ngf * 4),
        # nn.ReLU(True),
                

        ResizeConvolution(512, 256),
        # nn.BatchNorm2d(512),

        ResizeConvolution(256, 128),
        # nn.BatchNorm2d(512),
        # nn.GELU(),

        ResizeConvolution(128, 64),
        # nn.BatchNorm2d(32),
        # nn.GELU(),
        
        ResizeConvolution(64, 64),

        ResizeConvolution(64, 64),

        ResizeConvolution(64, 3),
        # ResizeConvolution(8, 3),

        # nn.ConvTranspose2d(8, 3, 3, 1, 1, bias=False),
        nn.Upsample(size=(config.img_size, config.img_size)),
        nn.Sigmoid()
    )

# decoder = nn.Sequential(
#     nn.ConvTranspose2d(NUM_VOXELS, ngf * 16, kernel_size=4, stride=2, padding=0, bias=False),
#     nn.BatchNorm2d(ngf * 16),
#     nn.ReLU(True),
#     nn.ConvTranspose2d(ngf * 16, ngf * 8, kernel_size=4, stride=2, padding=1, bias=False),
#     nn.BatchNorm2d(ngf * 8),
#     nn.ReLU(True),
#     nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1, bias=False),
#     nn.BatchNorm2d(ngf * 4),
#     nn.ReLU(True),
#     nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
#     nn.BatchNorm2d(ngf * 2),
#     nn.ReLU(True),
#     nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=3, stride=2, padding=1, bias=False),
#     nn.BatchNorm2d(ngf),
#     nn.ConvTranspose2d(ngf, 3, kernel_size=3, stride=2, padding=1, bias=False),
#     nn.Upsample(size=(config.img_size, config.img_size))
# )

# use AnyCost GAN instead......

# GAN here:

# custom weights initialization called on netG and netD
# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         nn.init.normal_(m.weight.data, 0.0, 0.02)
#     elif classname.find('BatchNorm') != -1:
#         nn.init.normal_(m.weight.data, 1.0, 0.02)
#         nn.init.constant_(m.bias.data, 0)

# Generator Code

# class Generator(nn.Module):
#     def __init__(self, ngpu):
#         super(Generator, self).__init__()
#         ngf = 64
#         nz = NUM_VOXELS
#         nc = 3
#         self.main = nn.Sequential(
#             # input is Z, going into a convolution
#             nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
#             nn.BatchNorm2d(ngf * 8),
#             nn.ReLU(True),
#             # state size. (ngf*8) x 4 x 4
#             nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ngf * 4),
#             nn.ReLU(True),
#             # state size. (ngf*4) x 8 x 8
#             nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ngf * 2),
#             nn.ReLU(True),
#             # state size. (ngf*2) x 16 x 16
#             nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ngf),
#             nn.ReLU(True),
#             # state size. (ngf) x 32 x 32
#             nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
#             nn.Tanh()
#             # state size. (nc) x 64 x 64
#         )

#     def forward(self, input):
#         return self.main(input)

# class Discriminator(nn.Module):
#     def __init__(self, ngpu):
#         super(Discriminator, self).__init__()
#         ndf = 64
#         self.main = nn.Sequential(
#             # input is (nc) x 64 x 64
#             nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf) x 32 x 32
#             nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf * 2),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf*2) x 16 x 16
#             nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf * 4),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf*4) x 8 x 8
#             nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf * 8),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf*8) x 4 x 4
#             nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
#             nn.Sigmoid()
#         )

#     def forward(self, input):
#         return self.main(input)

def make_original_decoder(NUM_VOXELS):
    ngf = 64
    return nn.Sequential(
        nn.Linear(NUM_VOXELS, NUM_VOXELS // 4),
        nn.GELU(),
        nn.Linear(NUM_VOXELS // 4, NUM_VOXELS // 4),
        nn.GELU(),
        nn.Linear(NUM_VOXELS // 4, NUM_VOXELS // 4),
        einops.layers.torch.Rearrange('b N -> b N 1 1'),
        nn.ConvTranspose2d(NUM_VOXELS // 4, ngf * 16, kernel_size=4, stride=2, padding=0, bias=False),
        nn.Upsample(scale_factor = 2),
        nn.BatchNorm2d(ngf * 16),
        nn.GELU(),
        nn.ConvTranspose2d(ngf * 16, ngf * 8, kernel_size=4, stride=2, padding=1, bias=False),
        nn.Upsample(scale_factor = 2),
        nn.BatchNorm2d(ngf * 8),
        nn.GELU(),
        nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1, bias=False),
        nn.Upsample(scale_factor = 2),
        nn.BatchNorm2d(ngf * 4),
        nn.GELU(),
        nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
        nn.Upsample(scale_factor = 2),
        nn.BatchNorm2d(ngf * 2),
        nn.GELU(),
        nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=3, stride=1, padding=1, bias=False),
        nn.Upsample(scale_factor = 2),
        nn.BatchNorm2d(ngf),
        nn.Upsample(scale_factor = 2),
        nn.Conv2d(ngf, 3, kernel_size=3, stride=1, padding=1, bias=False),
        nn.Upsample(size=(config.img_size, config.img_size)),
        nn.Sigmoid()
    )

class NewDecoderBlock(nn.Module):
    def __init__(self, in_channels):
        super(NewDecoderBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
        )
    
    def forward(self, x):
        return self.layers(x)



def new_decoder(NUM_VOXELS):
    # 2**8
    # ngf = 4**8
    depth = 6
    layers = [
        nn.Linear(NUM_VOXELS, 4**depth),
        einops.layers.torch.Rearrange('b N -> b N 1 1')
    ]
    for i in range(1, depth-1):
        d = depth - i
        layers.extend([
            nn.PixelShuffle(2),
            ResidualBlock(NewDecoderBlock(4**d)),
            ResidualBlock(NewDecoderBlock(4**d)),
            ResidualBlock(NewDecoderBlock(4**d)),
            ResidualBlock(NewDecoderBlock(4**d)),
            nn.BatchNorm2d(4**d)
            ]
    )
    layers.extend([
        nn.PixelShuffle(2),
        # ResizeConvolution(4**d, 4**d),
        # ResizeConvolution(4**d, 4**d),
        nn.Conv2d(4**(d-1), 3, kernel_size=3, stride=1, padding=1, bias=False),
        nn.Upsample(size=(config.img_size, config.img_size)),
        nn.Sigmoid()
    ])
    print(layers)
    return nn.Sequential(*layers)

        # nn.PixelShuffle(2),
        # ResidualBlock(NewDecoderBlock(4**8)),
        # ResidualBlock(NewDecoderBlock(4**8)),
        # nn.PixelShuffle(2),
        # ResidualBlock(NewDecoderBlock(4**7)),
        # ResidualBlock(NewDecoderBlock(4**7)),
        # nn.PixelShuffle(2),
        # ResidualBlock(NewDecoderBlock(4**6)),
        # ResidualBlock(NewDecoderBlock(4**6)),
        # nn.PixelShuffle(2),
        # ResidualBlock(NewDecoderBlock(4**6)),
        # ResidualBlock(NewDecoderBlock(4**6)),
            # nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(1024),
            # nn.GELU(),
            # nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(1024),
            # nn.GELU(),
            # nn.PixelShuffle(2),
            # nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(256),
            # nn.GELU(),
            # nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(256),
            # nn.GELU(),
            # nn.PixelShuffle(2),
            # nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(64),
            # nn.GELU(),
            # nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(64),
            # nn.GELU(),
            # nn.PixelShuffle(2),
        # nn.GELU(),
        # nn.PixelShuffle(2),
        # nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1, bias=False),
        # nn.BatchNorm2d(4),
        # nn.GELU(),
        # nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1, bias=False),
        # nn.BatchNorm2d(4),
        # nn.GELU(),
        # nn.PixelShuffle(2),

    # )