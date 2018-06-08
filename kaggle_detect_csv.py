import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
#sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
#import coco
# Import kaggle config
sys.path.append(os.path.join(ROOT_DIR, "samples/kaggle/"))  # To find local version
import kaggle

#%matplotlib inline

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
#COCO_MODEL_PATH = os.path.join(ROOT_DIR, "logs/kaggle20180608T0904/mask_rcnn_kaggle_0010.h5")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_kaggle_0020.h5")
print("COCO_MODEL_PATH = ", COCO_MODEL_PATH)

# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    print("COCO_MODEL_PATH NOT FOUND!")
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "test_2")
print("IMAGE_DIR", IMAGE_DIR)

class InferenceConfig(kaggle.KaggleConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'car', 'motorbicycle','bicycle','person','truck','bus','tricycle']

def rle_encode(x):
    """Encodes a mask in Run Length Encoding (RLE).
        Returns a string of space-separated values.
        """
    assert x.dtype == np.bool
    dots = np.where(x.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if b > prev + 1:
            run_lengths.append([b, 0])
        run_lengths[-1][1] += 1
        prev = b
    return '|'.join('{} {}'.format(*pair) for pair in run_lengths)

# Load a random image from the images folder
file_names = next(os.walk(IMAGE_DIR))[2]

# Generate class dict
class_dict = {0:0, 1:33, 2:34, 3:35, 4:36, 5:38, 6:39, 7:40}

f = open("prediction.csv", 'w')
f.write("ImageId,LabelId,Confidence,PixelCount,EncodedPixels\n")

for image_filename in file_names:
    #print("image = ", image_filename)
    image = skimage.io.imread(os.path.join(IMAGE_DIR,image_filename))
    image_id = image_filename[:-4]
    #print("image_id = ", image_id)
    # Run detection
    results = model.detect([image], verbose=1)
    
    #print("GOT RESULTS!!")
    # Visualize results
    r = results[0]
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],class_names, r['scores'])
    
    class_ids = r['class_ids']
    confidence = r['scores']
    masks = r['masks']
    
    # save image
    submit_dir = os.path.join(ROOT_DIR, "samples/kaggle/")
    plt.savefig("{}/{}".format(submit_dir, image_filename))
    
    #print("class_ids = ",class_ids)
    for i in range(0,len(class_ids)):
        if(class_ids[i] in class_dict.keys()):
            f.write(image_id+",")
            f.write(str(class_dict[class_ids[i]]) +",")
            f.write(str(confidence[i])+",")
            mask = masks[:,:,i]
            f.write(str(np.count_nonzero(mask)) +",")
            f.write(str(rle_encode(mask)))
            f.write('\n')

f.close()




