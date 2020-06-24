import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
#import sdfRender
from dfr.render import dfrRender
import torchvision
from im2mesh.onet.models import decoder
from im2mesh.common import normalize_imagenet

class Resnet18_rgb(nn.Module):
    r''' ResNet-18 encoder network for image(rgb) input.
    Args:
        c_dim (int): output dimension of the latent embedding
        normalize (bool): whether the input images should be normalized
        use_linear (bool): whether a final linear layer should be used
    '''

    def __init__(self, c_dim, normalize=True, use_linear=True):
        super().__init__()
        self.normalize = normalize
        self.use_linear = use_linear
        self.features = torchvision.models.resnet18(pretrained=True)
        self.features.fc = nn.Sequential()
        if use_linear:
            self.fc = nn.Linear(512, c_dim)
        elif c_dim == 512:
            self.fc = nn.Sequential()
        else:
            raise ValueError('c_dim must be 512 if use_linear is False')

    def forward(self, x):
        if self.normalize:
            x = normalize_imagenet(x)
        net = self.features(x)
        out = self.fc(net)
        return out

class Resnet18_alpha(nn.Module):
    r''' ResNet-18 encoder network for image(alpha) input.
    Args:
        c_dim (int): output dimension of the latent embedding
        normalize (bool): whether the input images should be normalized
        use_linear (bool): whether a final linear layer should be used
    '''

    def __init__(self, c_dim, normalize=True, use_linear=True):
        super().__init__()
        self.normalize = normalize
        self.use_linear = use_linear
        self.features = torchvision.models.resnet18(pretrained=True)
        self.features.fc = nn.Sequential()
        self.features.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        if use_linear:
            self.fc = nn.Linear(512, c_dim)
        elif c_dim == 512:
            self.fc = nn.Sequential()
        else:
            raise ValueError('c_dim must be 512 if use_linear is False')

    def forward(self, x):
        net = self.features(x)
        out = self.fc(net)
        return out
    
class reconstructor(nn.Module):
    '''predict -1(inside),1(outside) values of spatial point, conditioned with input image
    Args:
        c_dim(int) : condition dim(image features dim)
        rgb_alpha: the type of input image, true: rgb, false: alpha
        
        ######
        image_resolution(int): the rendered resolution of image
        
    
    '''
    def __init__(self, c_dim=128, rgb_alpha = True, image_resolution = 64, ray_steps = 32,
                 norm_method = 'batch_norm',random_sampling = False,unhit_avg = True,render_alpha = 20,
                 sample_neighborhood = False, neighborhood_points_num = 3, neighborhood_radius = 0.5, neighborhood_weight = 1,
                 random_unhit = False,mgpu = False        
                 ):
        super(reconstructor,self).__init__()
        self.c_dim = c_dim
        self.rgb_alpha = rgb_alpha
        if rgb_alpha:
            self.encoder = Resnet18_rgb(c_dim)
        else:
            self.encoder = Resnet18_alpha(c_dim)
            
        self.decoder = decoder.DecoderCBatchNorm(dim=3, z_dim=0, c_dim=128,
                 hidden_size=256, leaky=False, legacy=False,norm_method = norm_method)#instance_norm,group_norm)
        
        self.renderer = dfrRender.dfrRender(image_size=image_resolution,anti_aliasing=False,focal_length=1,steps = ray_steps,distance=2.732,bounding_sphere_radius=1.212,
                 image_length=1.0,random_sampling = random_sampling, unhit_avg = unhit_avg,render_alpha=render_alpha,
                 sample_neighborhood = sample_neighborhood, neighborhood_points_num = neighborhood_points_num, neighborhood_radius = neighborhood_radius, neighborhood_weight = neighborhood_weight,
random_unhit = random_unhit,mgpu =mgpu 
                 )
    def predict(self,points,z,c):
        result = self.decoder(points, z, c)
        length = torch.norm(points[0],dim=1)
        result[0,(length>self.renderer.bounding_sphere_radius).nonzero()] = 1
        return result
#    def predict(self, imgs):

class generator(nn.Module):
    '''predict -1(inside),1(outside) values of spatial point from input code z
    Args:
        z_dim(int) : dim of noise code z
        ######
        image_resolution(int): the rendered resolution of image

    
    '''
    def __init__(self, z_dim=128, image_resolution = 64, ray_steps = 32,
                 norm_method = 'batch_norm',random_sampling = False,unhit_avg = True,render_alpha = 20):
        super(generator,self).__init__()
        self.z_dim = z_dim
        self.decoder = decoder.Decoder(dim=3, z_dim=128, c_dim=0,
                 hidden_size=256, leaky=False)
        
        self.renderer = dfrRender.dfrRender(image_size=image_resolution,anti_aliasing=False,focal_length=1,steps = ray_steps,distance=2.732,bounding_sphere_radius=1.212,
                 image_length=1.0,random_sampling = random_sampling, unhit_avg = unhit_avg,render_alpha=render_alpha)
    def predict(self,points,z,c=None):
        result = self.decoder(points, z, c)
        length = torch.norm(points[0],dim=1)
        result[0,(length>self.renderer.bounding_sphere_radius).nonzero()] = 1
        return result

class NormalDiscriminator(nn.Module):
    def __init__(self, img_size=64):
        super(NormalDiscriminator,self).__init__()

        self.img_size = img_size 
        
        self.convs1 = nn.Conv2d(in_channels = 1, out_channels = 64, kernel_size=4, stride=2, padding=1)
        self.convs2 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size=4, stride=2, padding=1)
        self.convs3 = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size=4, stride=2, padding=1)
        self.convs4 = nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size=4, stride=2, padding=1)
        self.convs5 = nn.Conv2d(in_channels = 512, out_channels = 1, kernel_size=4, stride=2, padding=0)
        self.m = nn.AdaptiveAvgPool2d(1)
#        self.linear_v = nn.Linear(3,32,bias=False)
    def forward(self, imgs):    
        out = imgs.view(-1,1,self.img_size,self.img_size)
        out = F.relu(self.convs1(out))
#        view_feature = F.relu(self.linear_v(view))
#        h = torch.cat((image_feature,view_feature.view(-1,32,1,1).repeat(1,1,32,32)),1)
        out = F.relu(self.convs2(out))
        out = F.relu(self.convs3(out))
        out = F.relu(self.convs4(out))
        out = self.convs5(out)
        out = self.m(out)
        return out
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)
        
class discriminator(nn.Module):
    def __init__(self, img_size=64):
        super(discriminator,self).__init__()

        self.img_size = img_size 
        
        self.convs1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size=4, stride=2, padding=1)
        self.convs2 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size=4, stride=2, padding=1)
        self.convs3 = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size=4, stride=2, padding=1)
        self.convs4 = nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size=4, stride=2, padding=1)
        self.convs5 = nn.Conv2d(in_channels = 512, out_channels = 1, kernel_size=4, stride=2, padding=0)
    
        self.linear_v = nn.Linear(3,32,bias=False)
    def forward(self, imgs, view):    
        out = imgs.view(-1,1,64,64)
        image_feature = F.relu(self.convs1(out))
        view_feature = F.relu(self.linear_v(view))
        h = torch.cat((image_feature,view_feature.view(-1,32,1,1).repeat(1,1,32,32)),1)
        out = F.relu(self.convs2(h))
        out = F.relu(self.convs3(out))
        out = F.relu(self.convs4(out))
        out = self.convs5(out)
        return out
    
    def extract_feature(self,imgs,view):
        batch_size = imgs.shape[0]
        pool = torch.nn.AdaptiveAvgPool2d(2)
        out = imgs.view(-1,1,64,64)
        image_feature = F.relu(self.convs1(out))
        view_feature = F.relu(self.linear_v(view))
        h = torch.cat((image_feature,view_feature.view(-1,32,1,1).repeat(1,1,32,32)),1)
        out1 = F.relu(self.convs2(h))
        out2 = F.relu(self.convs3(out1))
        out3 = F.relu(self.convs4(out2))
        
        feature = torch.cat((pool(out1),pool(out2),pool(out3)),1)        
        feature = feature.view(batch_size,-1).contiguous()
        return feature
        
        
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)


def main():
    en = Resnet18_rgb(128)
    i = torch.rand(1,3,64,64)
    print(en(i).shape)
    
if __name__ == '__main__':
    main()
