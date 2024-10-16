"""
Implementation taken from https://github.com/meliketoy/LSGAN.pytorch
"""
import torch.nn as nn

from gan_compare.training.networks.generation.base_generator import BaseGenerator


class Generator(BaseGenerator):
    def __init__(
        self,
        nz: int,
        ngf: int,
        nc: int,
        ngpu: int,
        leakiness: float,
        bias: bool = False,
        is_instance_norm_used: bool= False,
        **kwargs,
    ):
        super(Generator, self).__init__(
            nz=nz,
            ngf=ngf,
            nc=nc,
            ngpu=ngpu,
            leakiness=leakiness,
            bias=bias,
        )
        # important for norm layer config
        self.is_instance_norm_used = is_instance_norm_used

        # input : z
        # Generator will be consisted with a series of deconvolution networks

        self.layer1 = nn.Sequential(
            # input : z
            # Generator will be consisted with a series of deconvolution networks
            # Input size : input latent vector 'z' with dimension (nz)*1*1
            # Output size: output feature vector with (ngf*8)*4*4
            nn.ConvTranspose2d(
                in_channels=nz,
                out_channels=ngf * 8,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False,
            ),
            self.normalize(ngf * 8),
            nn.ReLU(True),
        )

        self.layer2 = nn.Sequential(
            # Input size : input feature vector with (ngf*8)*4*4
            # Output size: output feature vector with (ngf*4)*8*8
            nn.ConvTranspose2d(
                in_channels=ngf * 8,
                out_channels=ngf * 4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            self.normalize(ngf * 4),
            nn.ReLU(True),
        )

        self.layer3 = nn.Sequential(
            # Input size : input feature vector with (ngf*4)*8*8
            # Output size: output feature vector with (ngf*2)*16*16
            nn.ConvTranspose2d(
                in_channels=ngf * 4,
                out_channels=ngf * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            self.normalize(ngf * 2),
            nn.ReLU(True),
        )

        self.layer4 = nn.Sequential(
            # Input size : input feature vector with (ngf*2)*16*16
            # Output size: output feature vector with (ngf)*32*32
            nn.ConvTranspose2d(
                in_channels=ngf * 2,
                out_channels=ngf,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            self.normalize(ngf),
            nn.ReLU(True),
        )

        self.layer5 = nn.Sequential(
            # Input size : input feature vector with (ngf)*32*32
            # Output size: output image with (nc)*(image width)*(image height)
            nn.ConvTranspose2d(
                in_channels=ngf,
                out_channels=nc,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.Tanh()  # To restrict each pixels of the fake image to 0~1
            # Yunjey seems to say that this does not matter much
        )


    def normalize(self, num_features):
        if self.is_instance_norm_used:
            return nn.InstanceNorm2d(num_features=num_features)
        else:
            return nn.BatchNorm2d(num_features=num_features)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)

        return out


# if __name__ == '__main__':
#     net = Generator(
#         nz = 100,
#         ngf = 64,
#         nc = 1
#     )
#     print "Input(=z) : ",
#     print(torch.randn(128,100,1,1).size())
#     y = net(Variable(torch.randn(128,100,1,1))) # Input should be a 4D tensor
#     print "Output(batchsize, channels, width, height) : ",
#     print(y.size())
