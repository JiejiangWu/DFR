import torch
import torch.nn as nn
from dfr.utils import geo_utils
import torch.nn.functional as F
import math,sys
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

epsilon = 0.0005

Divide = False
Num = 1010700
use_surface = False
def divide_evaluate(sampled_points,z,conditional,net):
    N = Num//sampled_points.shape[0]
    result = torch.zeros(sampled_points.shape[0],sampled_points.shape[1]).to(device)
    
    times = math.ceil(sampled_points.shape[1]/N)
    
    for i in range(times):
        if not i == times:
            result[:,i*N:(i+1)*N] = net(sampled_points[:,i*N:(i+1)*N,:],z,conditional)
            torch.cuda.empty_cache()
        else:
            result[:,i*N:] = net(sampled_points[:,i*N:,:],z,conditional)
            torch.cuda.empty_cache()            
    return result
#    if sampled_points
    

def first_nonzero(x, axis=0):
    nonz = (x > 0)
    any_nonz, idx_first_nonz = ((nonz.cumsum(axis) == 1) & nonz).max(axis)
    idx_first_nonz[any_nonz == 0] = -1
    return idx_first_nonz
        
class dfrRender(nn.Module):
    def __init__(self, image_size=64,anti_aliasing=True,focal_length=1,steps = 64,distance=2.732,bounding_sphere_radius=1.212,
                 image_length=1.0,random_sampling = False,unhit_avg = True,render_alpha = 20,
                 
                 sample_neighborhood = False, neighborhood_points_num = 3, neighborhood_radius = 0.5, neighborhood_weight = 1,
                 
                 intensity_directional = 0.6,color_directional = [1,1,1],light_direction = [0,1,0],
                 intensity_ambient = 0.5,color_ambient = [1,1,1],
                 random_unhit = False, mgpu = False,
                 ):
        super(dfrRender, self).__init__()
        self.mgpu = mgpu
        self.image_size = image_size
        self.anti_aliasing = anti_aliasing
        self.distance = distance

        self.focal_length = 1
        self.steps = steps
        self.m = nn.AdaptiveAvgPool2d((image_size,image_size))
        self.alpha = render_alpha
        self.sigmoid = torch.nn.Sigmoid()
        self.render_img_size = image_size
        self.bounding_sphere_radius = bounding_sphere_radius
        self.image_length = image_length
        self.random_sampling = random_sampling
        self.unhit_avg = unhit_avg
        self.render_alpha = render_alpha
        self.random_unhit = random_unhit

        self.sample_neighborhood = sample_neighborhood
        self.neighborhood_points_num= neighborhood_points_num
        self.neighborhood_radius = neighborhood_radius
        self.neighborhood_weight = neighborhood_weight
        
        self.intensity_directional = intensity_directional
        color_directional = torch.Tensor(color_directional)
        light_direction = torch.nn.functional.normalize(torch.Tensor(light_direction),dim = 0)
        self.register_buffer('color_directional', color_directional)
        self.register_buffer('light_direction', light_direction)
        
        self.intensity_ambient = intensity_ambient
        color_ambient = torch.Tensor(color_ambient)
        self.register_buffer('color_ambient', color_ambient)
        
        eye = torch.Tensor([0, 0, 2.732])

        
        if self.anti_aliasing:
            self.render_img_size = self.image_size * 2
        else:
            self.render_img_size = self.image_size
    
        # uv matrix    
        uv = torch.zeros(2,self.render_img_size,self.render_img_size)
        for xcord in range(0,self.render_img_size):  
            for ycord in range(0,self.render_img_size):
                uv[0,xcord,ycord] = (2 * xcord + 1) / self.render_img_size - 1
                uv[1,xcord,ycord] = (2 * ycord + 1) / self.render_img_size - 1
        
        # interpolation matrix
        inter = torch.Tensor(2,self.steps)
        inter[0] = torch.linspace(1,0,self.steps)
        inter[1] = torch.linspace(0,1,self.steps)
        
        # pixel in range
        insected_idx = geo_utils.compute_pixels_in_sphere(torch.Tensor([bounding_sphere_radius]),
                                                            torch.Tensor([distance]),
                                                            torch.Tensor([focal_length]),self.render_img_size,image_length)
        insected_idx1d = insected_idx[:,0] * self.render_img_size + insected_idx[:,1]
            
        # down,right vec, lengthed 0.5
        down_base = torch.Tensor([0,-0.5,0])
        right_base = torch.Tensor([0.5,0,0])
        down = torch.Tensor([0,-0.5,0])
        right = torch.Tensor([0.5,0,0])
        
        self.register_buffer('eye', eye)
        self.register_buffer('uv', uv)
        self.register_buffer('inter', inter)
        self.register_buffer('insected_idx', insected_idx)
        self.register_buffer('insected_idx1d', insected_idx1d)
        self.register_buffer('down_base', down_base)
        self.register_buffer('right_base', right_base)
        self.register_buffer('down', down)
        self.register_buffer('right', right)
    
        self.compute_insect_point()
    
    def look_at(self,distance,elevation,azimuth):
        batch = elevation.shape[0]
        self.eye = geo_utils.get_points_from_angles(distance,elevation,azimuth)
        self.down = geo_utils.get_vec_from_angles(elevation,azimuth,self.down_base.view(1,3).repeat(batch,1))
        self.right = geo_utils.get_vec_from_angles(elevation,azimuth,self.right_base.view(1,3).repeat(batch,1))
        
        insect1 = geo_utils.get_rotate_points_from_angles(elevation,azimuth,self.insect1_base.view(1,-1,3).repeat(batch,1,1))
        insect2 = geo_utils.get_rotate_points_from_angles(elevation,azimuth,self.insect2_base.view(1,-1,3).repeat(batch,1,1))
        self.register_buffer('insect1', insect1)
        self.register_buffer('insect2', insect2)
        return 0
    
    def adjust_sampling(self,random_sampling = True):
        self.random_sampling = random_sampling
    def adjust_batch(self,batch_size):
        device = self.eye.device
        self.eye = torch.zeros(batch_size,3).to(device)
        self.down = torch.zeros(batch_size,3,1).to(device)
        self.right = torch.zeros(batch_size,3,1).to(device)
    def adjust_steps(self,steps):
        self.steps = steps
        inter = torch.Tensor(2,self.steps)
        inter[0] = torch.linspace(1,0,self.steps)
        inter[1] = torch.linspace(0,1,self.steps)
        self.register_buffer('inter', inter)
        
    def adjust_resolution(self,image_resolution):
        self.image_size = image_resolution
        if self.anti_aliasing:
            self.render_img_size = self.image_size * 2
        else:
            self.render_img_size = self.image_size
        # uv matrix    
        uv = torch.Tensor(2,self.render_img_size,self.render_img_size).to(self.eye.device)
        for xcord in range(0,self.render_img_size):  
            for ycord in range(0,self.render_img_size):
                uv[0,xcord,ycord] = (2 * xcord + 1) / self.render_img_size - 1
                uv[1,xcord,ycord] = (2 * ycord + 1) / self.render_img_size - 1
        self.register_buffer('uv', uv)
        self.m = nn.AdaptiveAvgPool2d((self.image_size,self.image_size))
        
        insected_idx = geo_utils.compute_pixels_in_sphere(torch.Tensor([self.bounding_sphere_radius]),
#                                                    torch.Tensor([distance]),
                                                    self.distance, # TODO: assmues all input distances maintain same
                                                    torch.Tensor([self.focal_length]),self.render_img_size,self.image_length)
        insected_idx1d = insected_idx[:,0] * self.render_img_size + insected_idx[:,1]
        self.register_buffer('insected_idx', insected_idx)
        self.register_buffer('insected_idx1d', insected_idx1d)
        
        self.compute_insect_point()
    def compute_insect_point(self):
        # compute In_i and Out_i for each ray
        # also compute Length of eye to center_i
        
        if self.anti_aliasing:
            render_img_size = self.image_size * 2
        else:
            render_img_size = self.image_size
        
        
            
        batch = 1
        device = self.eye.device
        half_pixel_length = torch.Tensor([self.image_length/(render_img_size*2)]).to(device)
        down = torch.Tensor([0,-0.5,0]).to(device)
        right = torch.Tensor([0.5,0,0]).to(device)
        eye = torch.Tensor([0, 0, 2.732]).view(1,3).to(device)
        
        
        sphere_center = torch.Tensor([0,0,0]).to(device)
        radius = torch.Tensor([self.bounding_sphere_radius]).to(device)

        down_uv = self.uv[0].view(1,render_img_size,render_img_size,1).repeat(batch,1,1,1).view(batch,-1,1) # batch *(img_size)^2 * 1
                # batch *(img_size)^2 * 1             # batch * 1 * 3
        down_vec = torch.bmm(down_uv       ,     down.view(batch,-1,3)) # batch * (img_size)^2 * 3
        down_vec = down_vec.view(batch,render_img_size,render_img_size,3)
        
        right_uv = self.uv[1].view(1,render_img_size,render_img_size,1).repeat(batch,1,1,1).view(batch,-1,1)
        right_vec = torch.bmm(right_uv     ,     right.view(batch,-1,3))
        right_vec = right_vec.view(batch,render_img_size,render_img_size,3)
        
        cameraOrientation = -torch.nn.functional.normalize(eye,dim=1)       # batch * 3   
        cordInImage = eye.view(batch,1,1,3).repeat(1,render_img_size,render_img_size,1) +\
                        self.focal_length * cameraOrientation.view(batch,1,1,3).repeat(1,render_img_size,render_img_size,1) +\
                        down_vec+right_vec  # batch * img_size * img_size * 3


        forwardOrientation = cordInImage - eye.view(batch,1,1,3).repeat(1,render_img_size,render_img_size,1)   
        Eye2Center_length = torch.norm(forwardOrientation,dim=3)
        forwardOrientation = torch.nn.functional.normalize(forwardOrientation,dim=3)
        
        
#        selectedOrientation = torch.cat([forwardOrientation[x,y].unsqueeze(0) for x, y in zip(self.insected_idx[:,0], self.insected_idx[:,1])])
        
        selectedOrientation = torch.cat([forwardOrientation[:,x,y].unsqueeze(0) for x, y in zip(self.insected_idx[:,0], self.insected_idx[:,1])])  # select_len*batch*3
#        selectedOrientation2 = forwardOrientation[:,self.insected_idx[:,0],self.insected_idx[:,1],:]
        
        Eye2Center_length = Eye2Center_length[:,self.insected_idx[:,0],self.insected_idx[:,1]]
        selectedOrientation = selectedOrientation.transpose(0,1).contiguous()# batch*select_len*3
        

        insect1,insect2,insected=geo_utils.hit_sphere(sphere_center,radius,selectedOrientation,eye.view(batch,3))
        self.register_buffer('insect1_base', insect1)
        self.register_buffer('insect2_base', insect2)
        self.register_buffer('half_pixel_length',half_pixel_length)
        self.register_buffer('Eye2Center_length',Eye2Center_length)
        
        
    def render_silhouettes(self, SdfFunction, z, conditional,require_grad = True):
        if torch.cuda.device_count() > 1 and self.mgpu:
            SdfFunction = torch.nn.DataParallel(SdfFunction)
            SdfFunction = SdfFunction.cuda()
        
        if self.anti_aliasing:
            render_img_size = self.image_size * 2
        else:
            render_img_size = self.image_size
        
        if len(conditional)>0:
            batch = conditional.shape[0]
        elif len(z)>0:
            batch = z.shape[0]
            conditional = z
        else:
            print('Wrong input dimension of z or c, at least one non-empty')
            sys.exit()
        device = self.eye.device
        sphere_center = torch.Tensor([0,0,0]).to(device)
        radius = torch.Tensor([self.bounding_sphere_radius]).to(device)


        # result img
        img = torch.ones(batch,render_img_size,render_img_size,3).to(device)
        rayMarchingSDF = torch.ones(batch,render_img_size,render_img_size).to(device)
        
        '''ray emission'''
        
        insect1 = self.insect1.contiguous() # Ray In point
        insect2 = self.insect2.contiguous() # Ray out point
        
        ray_num = insect1.shape[1]
        steps = self.steps

        
        insect1 = insect1.view(-1,3)
        insect2 = insect2.view(-1,3)
        
        
        '''point evaluation with grad disabled'''
        if not self.random_sampling:
                        # (batch*select_line_num) *3*1    sample_len
            sampled_point = insect1.unsqueeze(2)*self.inter[0] + insect2.unsqueeze(2)*self.inter[1]# (batch*select_line_num) *3*sample_len
        else:
            random_offset = (torch.rand(self.steps) * (1/self.steps)).to(device)
            base0 = torch.linspace(0,1,self.steps+1).to(device)
            base0 = base0[0:-1]
            inter0 = base0+random_offset
            sampled_point = insect1.unsqueeze(2)*inter0 + insect2.unsqueeze(2)*(1-inter0)
        
        sampled_point = sampled_point.transpose(1,2) # (batch*select_line_num) *sample_len*3
                    
        sampled_point = sampled_point.contiguous().view(batch,ray_num*steps,3) # batch * (select_line_num*sample_len) * 3
        
        
        # 
        sampled_point_ = sampled_point.detach()
        conditional_ = conditional.detach()
        for p in SdfFunction.parameters():  # reset requires_grad
            p.requires_grad = False  
#        with torch.no_grad():   
        '''TODO divide'''  
        if not Divide:
            sdfValue = SdfFunction(sampled_point_,z,conditional_)# batch * (select_line_num*sample_len) * 1
        else:
            sdfValue = divide_evaluate(sampled_point_,z,conditional_,SdfFunction)
        sdfValue = sdfValue.view(batch,ray_num,steps)
        
        
        
        if use_surface:
            ''''''
            firstNegIdx = first_nonzero(sdfValue<0,2)  # batch * select_line_num
            ''''''
            if self.random_unhit:
                minSdf_ = torch.Tensor([1])
                minIdx = torch.randint(0,self.steps,[batch,ray_num]).to(device)
            else:
                minSdf_,minIdx = torch.min(sdfValue,2) # batch*ray_num
            
            ''''''
            codeNegIdx = (firstNegIdx == -1)  # hit ray
            codeMinIdx = 1 - codeNegIdx       # unhit ray
            finalSurfaceIdx = firstNegIdx*codeNegIdx.long() + minIdx*codeMinIdx.long()
            ''''''
            del sampled_point_,conditional_,minSdf_,sdfValue
            
            minIdx_offset = geo_utils.build_batch_index_offset(batch,ray_num,steps).to(device) # batch*ray_num
            
            ''''''
            minIdx = finalSurfaceIdx +minIdx_offset
            ''''''
        else:
            minSdf_,minIdx = torch.min(sdfValue,2) # batch*ray_num
            del sampled_point_,conditional_,minSdf_,sdfValue
            minIdx_offset = geo_utils.build_batch_index_offset(batch,ray_num,steps).to(device) # batch*ray_num
            minIdx = minIdx +minIdx_offset
        
        '''re-forward'''
        '''naive reforward all picked points'''
        if not self.sample_neighborhood:          
            sampled_point = sampled_point.view(batch*ray_num*steps,3)
            minIdx = minIdx.view(batch*ray_num,1)
            minSampled_point = sampled_point[minIdx]
            minSampled_point = minSampled_point.view(batch,ray_num,3)
            
            for p in SdfFunction.parameters():  # reset requires_grad
                p.requires_grad = True  
            minSdf = SdfFunction(minSampled_point,z,conditional)# (batch*ray_num) * 1
            minSdf = minSdf.view(-1)
            rayMarchingSDF = rayMarchingSDF.view(-1)
            batched_insected_idx1d = geo_utils.build_batch_select_line_index(self.insected_idx1d,render_img_size,batch)
            rayMarchingSDF[batched_insected_idx1d] = minSdf
            rayMarchingSDF=rayMarchingSDF.view(batch,render_img_size,render_img_size)
            
        else:
            sampled_point = sampled_point.view(batch*ray_num*steps,3)
            minIdx = minIdx.view(batch*ray_num,1)
            minSampled_point = sampled_point[minIdx]
            minSampled_point = minSampled_point.view(batch,ray_num,3)
            
            
#            Eye2Center_length
#            half_pixel_length
#            self.sample_neighborhood = sample_neighborhood
#            self.neighborhood_points_num= neighborhood_points_num
#            self.neighborhood_radius = neighborhood_radius
            
            length_eye_sampledpoint = torch.norm((minSampled_point-self.eye.view(batch,1,3)),dim=2)
            length_d = length_eye_sampledpoint/self.Eye2Center_length * self.half_pixel_length
            length_d *= self.neighborhood_radius
            

            '''sample neighbors' positions for each (surface or min) point'''            
                                  #batch *select rays *neighbor_num 
#            print(self.neighborhood_points_num)
            random_grow_direction = torch.rand(batch,ray_num,self.neighborhood_points_num,3).to(device)-0.5
            random_grow_direction = torch.nn.functional.normalize(random_grow_direction,dim =3)
            
            neighbors_points_offset = random_grow_direction*length_d.view(batch,ray_num,1,1)
            #batch *select rays *neighbor_num *3
            neighbors_points = neighbors_points_offset + minSampled_point.view(batch,ray_num,1,3) 
            
            #batch *(select rays *neighbor_num) *3
            neighbors_points = neighbors_points.view(batch,ray_num*self.neighborhood_points_num,3)
            
            for p in SdfFunction.parameters():  # reset requires_grad
                p.requires_grad = True              
            
            pickedPointsSdf = SdfFunction(minSampled_point,z,conditional)# (batch*ray_num) * 1
            neighPointsSdf = SdfFunction(neighbors_points,z,conditional)# (batch*(ray_num*neigh_num)*1)
            
            neighPointsSdf = neighPointsSdf.view(batch,ray_num,self.neighborhood_points_num)
            resultSdf = (neighPointsSdf.mean(2) * self.neighborhood_weight + pickedPointsSdf) / (self.neighborhood_weight+1)
            resultSdf = resultSdf.view(-1)
            rayMarchingSDF = rayMarchingSDF.view(-1)
            batched_insected_idx1d = geo_utils.build_batch_select_line_index(self.insected_idx1d,render_img_size,batch)
            rayMarchingSDF[batched_insected_idx1d] = resultSdf
            rayMarchingSDF=rayMarchingSDF.view(batch,render_img_size,render_img_size) 


        img = self.sigmoid(self.alpha*rayMarchingSDF)
        img_ = self.m(img)
        return img_
    
    def render_rgb(self, SdfFunction,z, conditional,require_grad = True):
        if self.anti_aliasing:
            render_img_size = self.image_size * 2
        else:
            render_img_size = self.image_size
            
        if len(conditional)>0:
            batch = conditional.shape[0]
        elif len(z)>0:
            batch = z.shape[0]
            conditional = z
        else:
            print('Wrong input dimension of z or c, at least one non-empty')
            sys.exit()
        device = self.eye.device
        sphere_center = torch.Tensor([0,0,0]).to(device)
        radius = torch.Tensor([self.bounding_sphere_radius]).to(device)


        # result img
        img = torch.ones(batch,render_img_size,render_img_size,3).to(device)
        rayMarchingSDF = torch.ones(batch,render_img_size,render_img_size).to(device)

        insect1 = self.insect1.contiguous() # Ray In point
        insect2 = self.insect2.contiguous() # Ray out point
        
        ray_num = insect1.shape[1]
        steps = self.steps

        
        insect1 = insect1.view(-1,3)
        insect2 = insect2.view(-1,3)
        
        if not self.random_sampling:
                        # (batch*select_line_num) *3*1    sample_len
            sampled_point = insect1.unsqueeze(2)*self.inter[0] + insect2.unsqueeze(2)*self.inter[1]# (batch*select_line_num) *3*sample_len
        else:
            random_offset = (torch.rand(self.steps) * (1/self.steps)).to(device)
            base0 = torch.linspace(0,1,self.steps+1).to(device)
            base0 = base0[0:-1]
            inter0 = base0+random_offset
            sampled_point = insect1.unsqueeze(2)*inter0 + insect2.unsqueeze(2)*(1-inter0)
        
        sampled_point = sampled_point.transpose(1,2) # (batch*select_line_num) *sample_len*3
                    
        sampled_point = sampled_point.contiguous().view(batch,ray_num*steps,3) # batch * (select_line_num*sample_len) * 3
        

        
        # 
        sampled_point_ = sampled_point.detach()
        conditional_ = conditional.detach()
        for p in SdfFunction.parameters():  # reset requires_grad
            p.requires_grad = False  
        sdfValue = SdfFunction(sampled_point_,z,conditional_)# batch * (select_line_num*sample_len)
        sdfValue = sdfValue.view(batch,ray_num,steps)
        
        ''''''
        firstNegIdx = first_nonzero(sdfValue<0,2)  # batch * select_line_num
        ''''''
        
        minSdf_,minIdx = torch.min(sdfValue,2) # batch*ray_num
        storeminIdx = minIdx.clone()########################################
        ''''''
        codeNegIdx = (firstNegIdx != -1)
        codeMinIdx = 1 - codeNegIdx
        finalSurfaceIdx = firstNegIdx*codeNegIdx.long() + minIdx*codeMinIdx.long()
        ''''''
        
        del sampled_point_,conditional_,minSdf_,sdfValue
        
        minIdx_offset = geo_utils.build_batch_index_offset(batch,ray_num,steps).to(device) # batch*ray_num
#        minIdx = minIdx +minIdx_offset
        ''''''
        minIdx = finalSurfaceIdx +minIdx_offset
        ''''''
        storeminIdx += minIdx_offset########################################

        
        sampled_point = sampled_point.view(batch*ray_num*steps,3)
        minIdx = minIdx.view(batch*ray_num,1)
        minSampled_point = sampled_point[minIdx]
        minSampled_point = minSampled_point.view(batch,ray_num,3)
        
        storeminIdx = storeminIdx.view(batch*ray_num,1)########################################
        storeSampled_point = sampled_point[storeminIdx]########################################
        storeSampled_point = storeSampled_point.view(batch,ray_num,3)########################################

#        for p in SdfFunction.parameters():  # reset requires_grad
#            p.requires_grad = True  
        
        ''''''
        upSdf_point = minSampled_point.clone()
        leftSdf_point = minSampled_point.clone()
        frontSdf_point = minSampled_point.clone()

#        downSdf_point = minSampled_point.clone()
#        rightSdf_point = minSampled_point.clone()
#        backSdf_point = minSampled_point.clone()
        
        leftSdf_point[:,:,0] += epsilon
        upSdf_point[:,:,1] += epsilon
        frontSdf_point[:,:,2] += epsilon
        
#        rightSdf_point[:,:,0] -= epsilon
#        downSdf_point[:,:,1] -= epsilon
#        backSdf_point[:,:,2] -= epsilon
        ''''''
#        leftSdf_point = leftSdf_point.detach()
#        upSdf_point = upSdf_point.detach()
#        frontSdf_point = frontSdf_point.detach()        
        if not require_grad:
            sampled_point = sampled_point.detach()
            leftSdf_point = leftSdf_point.detach()
            upSdf_point = upSdf_point.detach()
            frontSdf_point = frontSdf_point.detach()
            conditional = conditional.detach()

        minSdf = SdfFunction(minSampled_point,z,conditional)# (batch*ray_num) * 1
        minSdf = minSdf.view(-1)
        
        
        for p in SdfFunction.parameters():  # reset requires_grad
            p.requires_grad = False  
        storeminSdf = SdfFunction(storeSampled_point,z,conditional)# (batch*ray_num) * 1###############
        storeminSdf = storeminSdf.view(-1)#####################
        
        ''''''
        gradient = torch.zeros(batch,ray_num,3).to(device)
        upSdf = SdfFunction(upSdf_point,z,conditional)
        leftSdf = SdfFunction(leftSdf_point,z,conditional)
        frontSdf = SdfFunction(frontSdf_point,z,conditional)
        
        gradient[:,:,0] = (leftSdf-minSdf).squeeze()
        gradient[:,:,1] = (upSdf-minSdf).squeeze()
        gradient[:,:,2] = (frontSdf-minSdf).squeeze()
        surface_normal = torch.nn.functional.normalize(gradient,dim =2)
        
#        directional_light = F.relu(torch.bmm(self.light_direction.repeat(batch*ray_num,1).view(-1,1,3), surface_normal.view(-1, 3, 1)).view(batch,ray_num,1))
        
#        color = self.color_directional * directional_light * self.intensity_directional
        
#        color += self.color_ambient*self.intensity_ambient
        ''''''
        #surface normal map#
        color = surface_normal
#        res = torch.Tensor([-1,-1,-1]).to(device)
        color *= 0.5
        color += 0.5
                
        rayMarchingSDF = rayMarchingSDF.view(-1)
        batched_insected_idx1d = geo_utils.build_batch_select_line_index(self.insected_idx1d,render_img_size,batch)
        rayMarchingSDF[batched_insected_idx1d] = storeminSdf
        
        weight_alpha = self.sigmoid(1000*rayMarchingSDF)
        weight_alpha = 1-weight_alpha
        img = img.view(-1,3)
        img[batched_insected_idx1d] = color.view(-1,3)
        #img_ = img
        img_ = torch.bmm(img.view(-1,3,1),weight_alpha.view(-1,1,1))
        img_ = img_.view(batch,render_img_size,render_img_size,3).contiguous()
        img_ = img_.permute(0,3,1,2)
        
        img_ = self.m(img_)
        return img_    
    
