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
import kaggle2

#%matplotlib inline

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "logs/kaggle20180608T0904/mask_rcnn_kaggle_0012.h5")
print("COCO_MODEL_PATH = ", COCO_MODEL_PATH)

# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    print("COCO_MODEL_PATH NOT FOUND!")
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "train_color_15k")
print("IMAGE_DIR", IMAGE_DIR)

class InferenceConfig(kaggle2.KaggleConfig):
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

# Load a random image from the images folder
file_names = next(os.walk(IMAGE_DIR))[2]
#image_name = random.choice(file_names)
#image_name = "b35ca89ef593a6318e94cca715ac887c.jpg"
#image_name = "ff03c6490b209ee5515bc1a6cd4423bc.jpg"
#image_name = "171206_033656926_Camera_6.jpg"
#image_name = "171206_032652157_Camera_6.jpg"
image_name = "171206_032200927_Camera_5.jpg"
#image_name = "171206_030147787_Camera_5.jpg"
#image_name = "171206_034512629_Camera_5.jpg"
image = skimage.io.imread(os.path.join(IMAGE_DIR, image_name))
print("image_path = ", os.path.join(IMAGE_DIR, image_name))

# Run detection
results = model.detect([image], verbose=1)

# Visualize results
r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                            class_names, r['scores'])
submit_dir = os.path.join(ROOT_DIR, "samples/kaggle/")
print("image_name = ",image_name)
plt.savefig("{}/{}".format(submit_dir, image_name))
