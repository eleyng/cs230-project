import os
import sys
import json
import datetime
import numpy as np
import skimage.io
from skimage import data
from skimage.viewer import ImageViewer
#%matplotlib inline
from imgaug import augmenters as iaa
#from PyQt5 import QtCore, QtGui, QtWidgets
import matplotlib.pyplot as plt
# Root directory of the project
ROOT_DIR = os.path.abspath("../../")   ##### Modify it!

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize

from PIL import Image

def load_mask():
        
        #image_info = self.image_info[image_id]
        #print("IMAGE INFO", image_info)
        # if image_info["source"] != "coco":
        #     return super(CocoDataset, self).load_mask(image_id)

        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        #info = self.image_info[image_id]['id']
        #info = info[:-4] + '_instanceIds.png'
        # Get mask directory from image path
        # mask_dir = os.path.join(os.path.dirname(os.path.dirname(info['path'])), "masks")
        #mask_dir = os.path.join(ROOT_DIR, "train_label_10")
        # Read mask files from .png image
        
        class_ids = []
        mask = []
        #print('Image Id', image_id)
        
        idlist = [0,33,34,35,36,38,39,40]
        
        
        # Read image as array
        print("-"*50)
        
        m = skimage.io.imread("/Users/yarora/Downloads/Mask_RCNN-master/170908_061535211_Camera_5_instanceIds.png")
        
        
        class_id = m//1000
        
        class_bg_sub = np.zeros(m.shape)
        
        for id in idlist:
            if id == 0:
                continue
            else:
                print("ID!!!!!!", id)
                indices_obj = np.where(class_id==id)
                #print("indices_obj = ", indices_obj)
                if(len(indices_obj[0])>0):
                    class_obj = np.zeros(m.shape)
                    print("You are here with  id = ", id)
                    class_obj[indices_obj] = 1
                    mask.append(class_obj)
                    class_ids.append(id)
                    class_bg_sub = id*class_obj
                    #print("class_obj = ", class_obj)
                    
                    #viewer = ImageViewer(class_obj)
                    #viewer.show()
                    
            #visualize.display_images([class_obj])
        
        assert(False)
         
        class_bg = np.asarray(class_id - class_bg_sub).astype(np.bool)
        if np.count_nonzero(class_bg) > 0:
            mask.append(class_bg)
            class_ids.append(0)
        
        '''
        for f in next(os.walk(mask_dir))[2]:
            if f.endswith(".png"):
                
                
                m = skimage.io.imread(os.path.join(mask_dir, f)).astype(np.bool)
                mask.append(m)
                tlabel = np.asarray(Image.open(os.path.join(mask_dir, f)))
                class_id = tlabel//1000
                class_ids.append(class_id)
                print("CLASSID", class_id)
        '''
                
        mask = np.stack(mask, axis=-1)
        class_ids = np.array(class_ids, dtype=np.int32)
        #print("CLASS IDs", class_ids)
        
        return mask, class_ids

load_mask()
