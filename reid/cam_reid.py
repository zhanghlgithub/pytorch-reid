from __future__ import division, print_function, absolute_import
import torch
from torch.nn import functional as F
import sys
sys.path.insert(0, "/home/hlzhang/project/detection_reid/reid")
from torchreid import metrics
import cv2
from torchvision import transforms
from torch.autograd import Variable
import torchreid
import os
import numpy as np

def reid_model():
    model = torchreid.models.build_model(
        name="osnet_ain_x1_0",
        num_classes=100,
        loss="softmax",
        pretrained=True
    )
    
    if torch.cuda.is_available():          
        model = model.cuda()
    
    model.eval()
    
    return model
     
class Compare(object):
    
    def __init__(self, model=None,
                 origin_img="./data/origin_image", normalize_feature=True):
        '''args:
              - model_name: option diff model
              - origin_img: the path of base image
              - compaer_img: single image that want to compare with base image 
                             type: BGR           
        '''        
        self.model = model
        self.origin_img = origin_img
        self.is_normalize_f = normalize_feature
               
    def _extract_feature(self, model, input):
        '''input: BGR image'''
        
        #self.model.eval()
        
        input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
        input = cv2.resize(input, (128, 256)) 
        input = input.astype(np.float32) / 255.0  # add 2019.12.20
        input_as_tensor = transforms.ToTensor()(input)
        
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        input_as_tensor[0,:,:] = (input_as_tensor[0,:,:] - mean[0]) / std[0]
        input_as_tensor[1,:,:] = (input_as_tensor[1,:,:] - mean[1]) / std[1]
        input_as_tensor[2,:,:] = (input_as_tensor[2,:,:] - mean[2]) / std[2]
        input_as_tensor = input_as_tensor.unsqueeze(0)
        
        input_as_tensor = Variable(input_as_tensor)
        if torch.cuda.is_available():
            input_as_tensor = input_as_tensor.cuda()
        
        return self.model(input_as_tensor)
        
    def encode_origin_image(self):
        
        img_list = os.listdir(self.origin_img)
        f_, name_ = [], []
        for img_name in img_list:
            img_path = os.path.join(self.origin_img, img_name)
            img = cv2.imread(img_path)
            
            feature = self._extract_feature(self.model, img)
            feature = feature.data.cpu()
            
            f_.append(feature)
            name_.append(img_name.split(".")[0])
        
        f_ = torch.cat(f_, 0)
        
        if(self.is_normalize_f):
            f_ = F.normalize(f_, p=2, dim=1)
            
        return f_, name_     
    
    def run(self, compaer_img, origin_f, origin_name, dist_metric='cosine'):
        '''
            Args:
                - compaer_img: single image that want to compare with base image 
                             type: BGR          
        '''
        compare_f = self._extract_feature(self.model, compaer_img).data.cpu()
        
        if(self.is_normalize_f):
            #print('Normalzing features with L2 norm ...')
            compare_f = F.normalize(compare_f, p=2, dim=1)
            
        distmat = metrics.compute_distance_matrix(compare_f, origin_f, metric=dist_metric)
        distmat = distmat.numpy()
        dist_list = distmat.tolist()[0]   # to list
        
        #print("dist list:", dist_list, origin_name)
        
        #top_id = distmat.tolist()[0].index(min(distmat.tolist()[0]))
        top_id = dist_list.index(min(dist_list))
        if(min(dist_list) < 0.5):
            identify_name = origin_name[top_id]
        else:
            identify_name = "Unknow"
        
        return identify_name, min(dist_list)

        
def test():
    
    path1 = "./data/image"
    path2 = "./data/image/Zhang HL3.jpg"
    
    compare = Compare(model=reid_model(), origin_img=path1)
    origin_f, origin_name = compare.encode_origin_image()
    #print("origin_f dim:", origin_f.dim())
    
    compare_img = cv2.imread(path2)
    identify_name, score = compare.run(compare_img, origin_f, origin_name)
    print(identify_name, score)
    
if __name__=="__main__":  
    test()
        
    