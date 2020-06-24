import sys
sys.path.append("..")
from dfr.render import dfrRender
from dfr.utils import generator
import numpy as np
import torch
import imageio
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from skimage.io import imread


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class implicitfunction(nn.Module):
    def __init__(self):
        super(implicitfunction, self).__init__()
        self.linear1 = torch.nn.Linear(3,64)
        self.linear2 = torch.nn.Linear(64,128)
        self.linear3 = torch.nn.Linear(128,1)
        torch.nn.init.xavier_uniform(self.linear1.weight)
        torch.nn.init.constant(self.linear1.bias, 0.1)
        torch.nn.init.xavier_uniform(self.linear2.weight)
        torch.nn.init.constant(self.linear2.bias, 0.1)
        torch.nn.init.xavier_uniform(self.linear2.weight)
        torch.nn.init.constant(self.linear2.bias, 0.1)
        
    def forward(self,point,z,c):
        x = F.relu(self.linear1(point))
        x = F.relu(self.linear2(x))
        x = F.tanh(self.linear3(x))
        return x.squeeze(1)
    
class Model(nn.Module):
    def __init__(self,fit,renderer):
        super(Model, self).__init__()
        self.decoder = fit
        self.renderer = renderer
    def predict(self,points,z,c):
        result = self.decoder(points, z, c)
        length = torch.norm(points[0],dim=1)
        result[0,(length>self.renderer.bounding_sphere_radius).nonzero()] = 1
        result = result.squeeze()
        return result

def main():
    # define loss
    BCE = torch.nn.BCELoss().to(device)
    
    # define network
    net = implicitfunction().to(device)
    
    
    # define render
    render = dfrRender.dfrRender(anti_aliasing=False,steps = 32)
    render.adjust_resolution(224)
    render.adjust_batch(1)
    render = render.to(device)
#    render = dfrRender.dfrRender(image_size = 224, random_sampling = True,anti_aliasing=False,steps = 32)
#    render.adjust_batch(1)
#    render.alpha = 10
    render = render.to(device)

    model = Model(net,render)
    
    distance = torch.Tensor([2.732]).to(device)
    e = torch.Tensor([0]).to(device)
    a = torch.Tensor([0]).to(device)
    render.look_at(distance,e,a)

    # define optimizer
    optimizer = optim.Adam(net.parameters(),lr=1e-3)

    # load input img
    input_img = imread('./input/input_test2.png') / 255.

    target_img = 1-input_img[:,:,3]

    target_img = torch.Tensor(target_img).to(device)
    conditional = torch.zeros(1,1)
    
    writer = imageio.get_writer('example2.gif', mode='I',fps=50)
    
    
    # optimize
    for i in range(600):
        print(str(i)+'/'+str(600))
        img = render.render_silhouettes(net,[],conditional)
        Loss = BCE(img[0,:,:],target_img)
        optimizer.zero_grad()
        Loss.backward()
        optimizer.step()  
        image2 = img[0].detach().cpu().numpy()
        image2 = np.tile(image2,[3,1,1]).transpose(1, 2, 0)
        writer.append_data((255*image2).astype(np.uint8))
    writer.close()
    
    # generate result mesh
    model_G = generator.Generator3D(model,device = device,simplify_nfaces = 10000)
    mesh = model_G.generate_mesh(torch.Tensor([]))
    mesh.export('./test2.off')    

    
if __name__ == '__main__':
    main()
