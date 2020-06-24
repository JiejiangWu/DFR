import torch
import numpy as np
import scipy.misc
import neural_renderer as nr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def render_img_from_mesh(mesh,e,a,d,file_name):
    renderer = nr.Renderer(camera_mode='look_at',image_size=512,background_color=[1,1,1],light_intensity_ambient=0.3)
    renderer = renderer.to(device)
    renderer.eye = nr.get_points_from_angles(d, e, a)   
    texture_size = 2
    v = torch.Tensor(mesh.vertices)[None, :, :]
    f = torch.Tensor(mesh.faces)[None, :, :]
    
    v=v.to(device)
    f = f.int().to(device)
    
    textures = torch.ones(1, f.shape[1], texture_size, texture_size, texture_size, 3, dtype=torch.float32).cuda()
    textures[:] = 0.9
    images,_,_ = renderer(v, f, textures)
    image1 = images.detach().cpu().numpy()[0].transpose(1, 2, 0)
    image2 = (image1*255).astype(np.uint8)
    scipy.misc.imsave(file_name, image2)
    return image1

def render_img_from_vf(vertices,faces,e,a,d,file_name):
    renderer = nr.Renderer(camera_mode='look_at',image_size=512,background_color=[1,1,1],light_intensity_ambient=0.3)
    renderer = renderer.to(device)
    renderer.eye = nr.get_points_from_angles(d, e, a)   
    texture_size = 2
    v = vertices[None, :, :]
    f = faces[None, :, :]
    
    v=v.to(device)
    f = f.int().to(device)
    
    textures = torch.ones(1, f.shape[1], texture_size, texture_size, texture_size, 3, dtype=torch.float32).cuda()
    textures[:] = 0.9
    images,_,_ = renderer(v, f, textures)
    image1 = images.detach().cpu().numpy()[0].transpose(1, 2, 0)
    image2 = (image1*255).astype(np.uint8)
    scipy.misc.imsave(file_name, image2)
    return image1


