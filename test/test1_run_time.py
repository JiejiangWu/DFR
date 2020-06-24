import sys
sys.path.append("..")
from dfr.render import dfrRender
from dfr.models import models
import numpy as np
import torch
import imageio
import time,os
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ss', type = int,default = 32)
    args = parser.parse_args()

    # define render
    render = dfrRender.dfrRender(image_size = 128, anti_aliasing=False,steps = args.ss)
    render.adjust_batch(1)
    render.alpha = 50
    render = render.to(device)

    
    # load pre-trained model
    model_G = models.generator(z_dim = 128, ray_steps = 32, random_sampling = True,render_alpha = 10)
    model_G.renderer.adjust_batch(16)
    checkpoint_dir = './checkpoints/gan-chair.pth.tar'
    if os.path.exists(checkpoint_dir):
        state = torch.load(checkpoint_dir)

        '''load G'''
        pre_G_dict = state['model_G']
        model_dict = model_G.state_dict()
        pretrained_dict = {k: v for k, v in pre_G_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        model_G.load_state_dict(model_dict)
        model_G = model_G.to(device)
    else:
        print('no checkpoint file')
        return
    # random vector
    z_noise=torch.empty(1,128).normal_(mean=0,std=.33)
    conditional = z_noise.to(device)

    e = 30
    for azimuth in range(0,360,90):
        writer = imageio.get_writer('./test1_silhouette'+str(azimuth)+'.png')
        time1 = time.time()
        render.look_at(torch.Tensor([2.732]).to(device),torch.Tensor([e]).to(device),torch.Tensor([azimuth]).to(device))
        img = render.render_silhouettes(model_G.decoder,conditional,[])
        print('silhouette frame elapsed time:')
        print(time.time()-time1)
        image2 = img[0].detach().cpu().numpy()
        image2 = np.tile(image2,[3,1,1]).transpose(1, 2, 0)
        writer.append_data((255*image2).astype(np.uint8))
        writer.close()     
    for azimuth in range(0,360,90):
        writer = imageio.get_writer('./test1_normalMap'+str(azimuth)+'.png')
        time1 = time.time()
        render.look_at(torch.Tensor([2.732]).to(device),torch.Tensor([e]).to(device),torch.Tensor([azimuth]).to(device))
        img = render.render_rgb(model_G.decoder,conditional,[])
        print('surface normal map frame elapsed time:')
        print(time.time()-time1)
        image2 = img.detach().cpu().numpy()[0]
        image2 = image2.transpose(1, 2, 0)
        writer.append_data((255*image2).astype(np.uint8))
        writer.close()     
if __name__ == '__main__':
    main()
