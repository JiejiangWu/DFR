import torch
import numpy as np
import math

def get_points_from_angles(distance, elevation, azimuth, degrees=True):
#    if isinstance(distance, float) or isinstance(distance, int):
#        if degrees:
#            elevation = math.radians(elevation)
#            azimuth = math.radians(azimuth)
#        return (
#            distance * math.cos(elevation) * math.sin(azimuth),
#            distance * math.sin(elevation),
#            -distance * math.cos(elevation) * math.cos(azimuth))
#    else:
        if degrees:
            elevation = -math.pi/180. * elevation
            azimuth = -math.pi/180. * azimuth
    #
        return torch.stack([
            -distance * torch.cos(elevation) * torch.sin(azimuth),
            -distance * torch.sin(elevation),
            distance * torch.cos(elevation) * torch.cos(azimuth)
            ]).transpose(1,0)
    
def get_vec_from_angles(elevation,azimuth,vec,degrees=True):
   #elevation: n
   #azimuth: n
   #vec: n*3
    device = elevation.device
    
    if degrees:
        elevation = -math.pi/180. * elevation
        azimuth = -math.pi/180. * azimuth
    
    batch = elevation.shape[0]
    
#    cosAzi = torch.cos(azimuth).squeeze()
#    sinAzi = torch.sin(azimuth).squeeze()
#    cosEle = torch.cos(elevation).squeeze()
#    sinEle = torch.sin(elevation).squeeze()
#    zero = torch.zeros(batch).to(device)
    
    rotation = torch.zeros(batch,3,3).to(device)

    rotation = torch.stack([
        torch.stack([torch.cos(azimuth),-torch.sin(elevation)*torch.sin(azimuth),-torch.cos(elevation)*torch.sin(azimuth)]).transpose(0,1), # batch * 3
        torch.stack([torch.zeros(batch).to(device),torch.cos(elevation),-torch.sin(elevation)]).transpose(0,1),
        torch.stack([torch.sin(azimuth),torch.sin(elevation)*torch.cos(azimuth),torch.cos(elevation)*torch.cos(azimuth)]).transpose(0,1)
        ]).transpose(0,1)
#    rotation = torch.cat([
#           torch.cat([torch.cos(azimuth),-torch.sin(elevation)*torch.sin(azimuth),-torch.cos(elevation)*torch.sin(azimuth)]),
#            torch.cat([torch.zeros(batch).view(azimuth.shape).to(device),torch.cos(elevation),-torch.sin(elevation)]),
#            torch.cat([torch.sin(azimuth),torch.sin(elevation)*torch.cos(azimuth),torch.cos(elevation)*torch.cos(azimuth)])
#            ])
    return torch.bmm(rotation.view(batch,3,3),vec.view(batch,3,1))

def get_rotate_points_from_angles(elevation,azimuth,points,degrees=True):
   #elevation: batch
   #azimuth: batch
   #vec: batch*m*3
    device = elevation.device
    
    if degrees:
        elevation = -math.pi/180. * elevation
        azimuth = -math.pi/180. * azimuth
    
    batch = elevation.shape[0]
    
#    cosAzi = torch.cos(azimuth).squeeze()
#    sinAzi = torch.sin(azimuth).squeeze()
#    cosEle = torch.cos(elevation).squeeze()
#    sinEle = torch.sin(elevation).squeeze()
#    zero = torch.zeros(batch).to(device)
    
    rotation = torch.zeros(batch,3,3).to(device)

    rotation = torch.stack([
        torch.stack([torch.cos(azimuth),-torch.sin(elevation)*torch.sin(azimuth),-torch.cos(elevation)*torch.sin(azimuth)]).transpose(0,1), # batch * 3
        torch.stack([torch.zeros(batch).to(device),torch.cos(elevation),-torch.sin(elevation)]).transpose(0,1),
        torch.stack([torch.sin(azimuth),torch.sin(elevation)*torch.cos(azimuth),torch.cos(elevation)*torch.cos(azimuth)]).transpose(0,1)
        ]).transpose(0,1)
#    rotation = torch.cat([
#           torch.cat([torch.cos(azimuth),-torch.sin(elevation)*torch.sin(azimuth),-torch.cos(elevation)*torch.sin(azimuth)]),
#            torch.cat([torch.zeros(batch).view(azimuth.shape).to(device),torch.cos(elevation),-torch.sin(elevation)]),
#            torch.cat([torch.sin(azimuth),torch.sin(elevation)*torch.cos(azimuth),torch.cos(elevation)*torch.cos(azimuth)])
#            ])
    points = points.transpose(1,2)#batch,3,m
    return torch.bmm(rotation.view(batch,3,3),points).transpose(1,2)
def rotate_m_for_ea(elevation,azimuth):
    roll = elevation
    pitch = -azimuth
    yaw = 0
    a = np.deg2rad(yaw)
    b = np.deg2rad(pitch)
    c = np.deg2rad(roll)
    
    M = np.zeros([3,3])
    ca = np.cos(a)
    sa = np.sin(a)
    cb = np.cos(b)
    sb = np.sin(b)
    cc = np.cos(c)
    sc = np.sin(c)
    
    M[0,0]= ca*cb
    M[0,1]= ca*sb*sc-sa*cc
    M[0,2]= ca*sb*cc+sa*sc
    M[1,0]= sa*cb
    M[1,1]= sa*sb*sc+ca*cc
    M[1,2]= sa*sb*cc-ca*sc
    M[2,0]= -sb
    M[2,1]= cb*sc
    M[2,2]= cb*cc
    
    return M

def rotate_M_from_ypr(yaw, pitch, roll, order = 'xyz'):
    a = np.deg2rad(yaw)
    b = np.deg2rad(pitch)
    c = np.deg2rad(roll)
    
    
    ca = np.cos(a)
    sa = np.sin(a)
    cb = np.cos(b)
    sb = np.sin(b)
    cc = np.cos(c)
    sc = np.sin(c)
    
    # rotate along z, yaw
    Mz = np.array([
            [ca ,-sa,  0],
            [sa , ca,  0],
            [0  ,  0,  1]
            ])
    
    My = np.array([
            [cb ,  0, sb],
            [0  ,  1,  0],
            [-sb,  0, cb]
            ])
    
    Mx = np.array([
            [1  ,  0,  0],
            [0  , cc,-sc],
            [0 ,  sc, cc]
            ])
    
    M = {'x':Mx,'y':My,'z':Mz}
    m = np.array([
            [1  ,  0,  0],
            [0  ,  1,  0],
            [0  ,  0,  1]
            ])
    
    for i in order:
        m = np.dot(M[i],m)
    return m
def hit_sphere_old(sphere_center, radius, ray_direction, ray_origin):
    # sphere_center: 1*3 tensor
    # radius: 1*1 tensor
    # ray_direction: batch*3 tensor
    # ray_origin: batch*3 tensor
    batch_size = ray_direction.shape[0]
    sphere_center = sphere_center.repeat(batch_size,1)
    oc = ray_origin - sphere_center
    a = (ray_direction*ray_direction).sum(-1)
    b = 2.0*(oc*ray_direction).sum(-1)
    c = (oc*oc).sum(-1) - radius*radius
    discriminant = b*b - 4*a*c    
    
    sqrt = torch.sqrt(discriminant)
    
    insect1 = (-b-sqrt)/(2.0*a)
    insect2 = (-b+sqrt)/(2.0*a)
    
    insected = sqrt > 0
    
    return insect1.unsqueeze(1)*ray_direction+ray_origin,insect2.unsqueeze(1)*ray_direction+ray_origin,insected

def hit_sphere(sphere_center, radius, ray_direction, ray_origin):
    # sphere_center: 1*3 tensor
    # radius: 1*1 tensor
    # ray_direction: batch*selected_line_num*3 tensor
    # ray_origin: batch*3 tensor
    

    batch_size = ray_direction.shape[0]
    line_num = ray_direction.shape[1]
    
    ray_origin = ray_origin.view(batch_size,1,3).repeat(1,line_num,1)
    
    ray_direction = ray_direction.view(batch_size*line_num,3)
    ray_origin = ray_origin.view(batch_size*line_num,3)
    
    
    sphere_center = sphere_center.repeat(batch_size*line_num,1)
    oc = ray_origin - sphere_center
    a = (ray_direction*ray_direction).sum(-1)
    b = 2.0*(oc*ray_direction).sum(-1)
    c = (oc*oc).sum(-1) - radius*radius
    discriminant = b*b - 4*a*c    
        
    sqrt = torch.sqrt(discriminant)
    

    
    delta1 = (-b-sqrt)/(2.0*a)
    delta2 = (-b+sqrt)/(2.0*a)
    
    insected = sqrt > 0
    

    
    insect1 = delta1.unsqueeze(1)*ray_direction+ray_origin
    insect2 = delta2.unsqueeze(1)*ray_direction+ray_origin    
    insect1 = insect1.view(batch_size,line_num,3).contiguous()
    insect2 = insect2.view(batch_size,line_num,3).contiguous()
    insected = insected.view(batch_size,line_num).contiguous()

    
    return insect1, insect2,insected
        # batch * select_line_num * 3, batch * select_line_num * 3, batch * select_line_num  


def compute_pixels_in_sphere(radius,distance,focal_length,image_resolution,image_length=1.0):
    # radius: 1*1 tensor, the radius of bounding sphere, centered at origin point
    # distance: 1*1 tensor, the distance between eye and origin point
    # focal_length: 1*1 tensor
    # image_resolution: int
    # image_length: float, the length of projected plane
    # return: 2*n tensor, index of hitted piexls

    tangent_line_len = torch.sqrt(distance*distance-radius*radius)
    hit_range_in_image = radius*focal_length/tangent_line_len
    
#    down = torch.Tensor([0,-image_length/2])
#    right = torch.Tensor([image_length/2,0])
    center = torch.Tensor([image_length/2,-image_length/2])
    
    cordPerPixel = torch.zeros(image_resolution,image_resolution,2)
    for xcord in range(0,image_resolution):
        for ycord in range(0,image_resolution):
            cordPerPixel[xcord,ycord,0] = (0.5 + xcord) / (image_resolution) * image_length
            cordPerPixel[xcord,ycord,1] = -(0.5 + ycord) / (image_resolution) * image_length
    resCordPerPixel = cordPerPixel - center
    disPerPixel = torch.norm(resCordPerPixel,dim=2)
    
    return (disPerPixel <= hit_range_in_image).nonzero()

def build_batch_index_offset(batch,ray_num,steps):
    m = torch.arange(0,ray_num,1)*steps
    m = m.repeat(batch,1)
    b = torch.arange(0,batch,1)*ray_num*steps
    b = b.view(-1,1).contiguous()
    b = b.repeat(1,ray_num)
    m = m+b
    return m

def build_batch_select_line_index(selected_idx1d,image_resolution,batch):
    # selected_idx1d : selected ray(pixel) index: n*1
    # image_resolution: int, the total pixels count (image_resolution*image_resolution)
    # return: batched selected_idx
    # example: 
    # selected_idx1d: 2,3,4, image_reolustion:3 (0,1,2,3,4,5,6,7,8), batch:3
    # output: 2,3,4,11,12,13,20,21,22
    ray_num = selected_idx1d.shape[0]
    device = selected_idx1d.device
    base = torch.arange(0,batch).to(device)
    base = base.view(batch,1).repeat(1,ray_num)*image_resolution*image_resolution
#    base = base.view(-1)
#    selected_idx1d.view(1,-1).repeat(batch,1)
    return (selected_idx1d+base).view(-1).contiguous()
    
def main():
#    test hit_sphere
#    s = torch.Tensor([[0,0,0]])
#    r = torch.Tensor([1])
#    rd = torch.Tensor([[1,1,1]])
#    rd = torch.nn.functional.normalize(rd)
#    ro = torch.Tensor([1,0,0])
#    i1,i2,id=hit_sphere(s,r,rd,ro)
#    print(i1)
#    print(i2)
#    print(id)
#    
    # test compute_pixels_hit_sphere
#    r = torch.Tensor([1])
#    distance = torch.Tensor([2.732])
#    focal_length = torch.Tensor([1])
#    print(compute_pixels_hit_sphere(r,distance,focal_length,64,1.0))
    
    # test get_vec_from_angles
    e = torch.Tensor([0,0,0,0,0])
    a = torch.Tensor([90,0,0,0,0])
    vec = torch.Tensor([0,1,5]).view(1,3).repeat(5,1)
    print(get_vec_from_angles(e,a,vec).squeeze())
if __name__ == '__main__':
    main()