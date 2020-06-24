import sys
sys.path.append("..")
from dfr.render import dfrRender
from dfr.utils import generator
from dfr.models import models
import numpy as np
import torch
import imageio,os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from skimage.io import imread
import argparse



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser()    
    parser.add_argument('--c_path', type = str,default = './checkpoints/airplane.pth.tar')
    parser.add_argument('--g_num', type = int,default = 10)
    
    args = parser.parse_args()
    
    model_G = models.generator(z_dim = 128, ray_steps = 32, random_sampling = True,render_alpha = 10)
    model_G.renderer.adjust_batch(16)

    
    if os.path.exists(args.c_path):
        state = torch.load(args.c_path)

        '''load G'''
        pre_G_dict = state['model_G']
        model_dict = model_G.state_dict()
        pretrained_dict = {k: v for k, v in pre_G_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        model_G.load_state_dict(model_dict)
        model_G = model_G.to(device)


    generator_G = generator.Generator3D(model_G,device = device,simplify_nfaces = 10000)
    for i in range(args.g_num):
        ''''''''''''''''''
        '''random generation'''
        ''''''''''''''''''
        print(i)
        z_noise=torch.empty(1,128).normal_(mean=0,std=.33)
        mesh = generator_G.generate_mesh_from_z(z_noise)
        mesh.export('./gan/'+str(i)+'.off')
    
if __name__ == '__main__':
    main()
