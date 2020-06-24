import sys
sys.path.append("..")
from dfr.render import dfrRender
from dfr.utils import generator
from dfr.models import models
import numpy as np
import torch
import imageio,os
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from skimage.io import imread


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser()    
    parser.add_argument('--input_path', type = str,default = './reconstruction/input')
    parser.add_argument('--output_path', type = str,default = './reconstruction/output')
    
    args = parser.parse_args()
    
    batch_size = 24
    checkpoint_path = './checkpoints/reconstruction.pth.tar'
    
    model_R = models.reconstructor(norm_method = 'batch_norm',random_sampling = True,unhit_avg = False,ray_steps=32)
    model_R.renderer.adjust_batch(batch_size)
    
    model = torch.load(checkpoint_path)
    
    pre_dict = model['R']
    model_dict = model_R.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pre_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict) 
    # 3. load the new state dict
    #        model.load_state_dict(model_dict)
    
    model_R.load_state_dict(model_dict)
    model_R = model_R.to(device)

    model_R.eval()
    model_G = generator.Generator3D(model_R,device = device,simplify_nfaces = 10000)
 
    img_files = os.listdir(args.input_path)
    
    for i in range(len(img_files)):
        print(str(i) + '/' + str(len(img_files)))
        temp_img_path = args.input_path + '/' + img_files[i]
        filename, file_extension = os.path.splitext(img_files[i])
        
        if not (file_extension == '.png' or file_extension == '.jpg'):
            assert('not acceptable format:' + temp_img_path)
            
#        temp_img = misc.imread(temp_img_path)
        temp_img = imread(temp_img_path)
        temp_img = temp_img.transpose(2,0,1)
        img = torch.FloatTensor(temp_img / 255.).unsqueeze(0).to(device)
        mesh = model_G.generate_mesh(img)
        mesh.export(args.output_path + '/' + filename + '.off')


if __name__ == '__main__':
    main()