#https://github.com/matterport/Mask_RCNN/blob/master/samples/nucleus/nucleus.py
# https://engineering.matterport.com/splash-of-color-instance-segmentation-with-mask-r-cnn-and-tensorflow-7c761e238b46
# https://github.com/matterport/Mask_RCNN/blob/master/samples/coco/inspect_data.ipynb

#COMMAND TO RUN:
#python kaggle_new.py train --dataset ../../ --weights coco --subset train

if __name__ == '__main__':
    import matplotlib
    # Agg backend runs without a display
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

import os
import sys
import json
import datetime
import numpy as np
import skimage.io
from imgaug import augmenters as iaa
from sklearn.cross_validation import train_test_split  ## Newly imported (by Hubert)

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")   ##### Modify it!

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize

from PIL import Image

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Results directory
# Save submission files here
RESULTS_DIR = os.path.join(ROOT_DIR, "results/kaggle/")

# The dataset doesn't have a standard train/val split, so I picked
# a variety of images to surve as a validation set.

###### --> Validation set (Substitue hand-coding!)
# VAL_IMAGE_IDS = [
#     "171206_030441892_Camera_6.jpg"
# ]

# Function to map images to class ids
def mapImageToClasses(image_file):
    # image_file: path to the image
    image = np.asarray(Image.open(image_file))
    cls =  np.unique(image)//1000
    unique, counts = np.unique(cls, return_counts=True)




############################################################
#  Configurations
############################################################

class KaggleConfig(Config):
    """Configuration for training on the nucleus segmentation dataset."""
    # Give the configuration a recognizable name
    NAME = "kaggle"

    # Adjust depending on your GPU memory
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 8  # Background + nucleus
    
    IMAGE_META_SIZE = 21


    # Number of training and validation steps per epoch
    # STEPS_PER_EPOCH = (657 - len(VAL_IMAGE_IDS)) // IMAGES_PER_GPU
    # VALIDATION_STEPS = max(1, len(VAL_IMAGE_IDS) // IMAGES_PER_GPU)  # Not used!
    #STEPS_PER_EPOCH = 100

    # Don't exclude based on confidence. Since we have two classes
    # then 0.5 is the minimum anyway as it picks between nucleus and BG
    # DETECTION_MIN_CONFIDENCE = 0
    #DETECTION_MIN_CONFIDENCE = 0  # Randomly try


    # Backbone network architecture
    # Supported values are: resnet50, resnet101
    #BACKBONE = "resnet101"

    # Input image resizing
    # Random crops of size 512x512
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 1024
    IMAGE_MIN_SCALE = 2.0

    # Length of square anchor side in pixels
    #RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)

    # ROIs kept after non-maximum supression (training and inference)
    #POST_NMS_ROIS_TRAINING = 1000
    #POST_NMS_ROIS_INFERENCE = 2000

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    #RPN_NMS_THRESHOLD = 0.9

    # How many anchors per image to use for RPN training
    #RPN_TRAIN_ANCHORS_PER_IMAGE = 64

    # Image mean (RGB)
    #MEAN_PIXEL = np.array([43.53, 39.56, 48.22])

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    #TRAIN_ROIS_PER_IMAGE = 128

    # Maximum number of ground truth instances to use in one image
    #MAX_GT_INSTANCES = 40    ## 200 for nucleus

    # Max number of final detections per image
    #DETECTION_MAX_INSTANCES = 50   ## 400 for nucleus
    
    # Steps per Epoch
    # STEPS_PER_EPOCH = 1000
    STEPS_PER_EPOCH = 5

    
 


class KaggleInferenceConfig(KaggleConfig):
    # Set batch size to 1 to run one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Don't resize imager for inferencing
    IMAGE_RESIZE_MODE = "pad64"
    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7


############################################################
#  Dataset
############################################################

class KaggleDataset(utils.Dataset):

    def load_kaggle(self, dataset_dir, subset):
        """Load a subset of the nuclei dataset.
        dataset_dir: Root directory of the dataset
        subset: Subset to load. Either the name of the sub-directory,
                such as stage1_train, stage1_test, ...etc. or, one of:
                * train: stage1_train excluding validation images
                * val: validation images from VAL_IMAGE_IDS
        """
        # Add classes. We have 8 classes plus one background class
        # The data we pre-processed should transform any other objet to "0"
#        self.add_class("kaggle", 33, "car")
#        self.add_class("kaggle", 34, "motorbicycle")
#        self.add_class("kaggle", 35, "bicycle")
#        self.add_class("kaggle", 36, "person")
#        self.add_class("kaggle", 38, "truck")
#        self.add_class("kaggle", 39, "bus")
#        self.add_class("kaggle", 40, "tricycle")
#        self.add_class("kaggle", 0, "others")

        self.add_class("kaggle", 1, "car")
        self.add_class("kaggle", 2, "motorbicycle")
        self.add_class("kaggle", 3, "bicycle")
        self.add_class("kaggle", 4, "person")
        self.add_class("kaggle", 5, "truck")
        self.add_class("kaggle", 6, "bus")
        self.add_class("kaggle", 7, "tricycle")
        #self.add_class("kaggle", 0, "others")
        
        
        # Which subset?
        # "val": use hard-coded list above
        # "train": use data from stage1_train minus the hard-coded list above
        # else: use the data from the specified sub-directory

        # assert subset in ["train", "val", "stage1_train", "stage1_test", "stage2_test"]
        # subset_dir = "stage1_train" if subset in ["train", "val"] else subset
        # dataset_dir = os.path.join(dataset_dir, subset_dir)

        assert subset in ["train", "val", "test"]    ## No test is split (if add test set, should be careful in radom shffuling)
        subset_dir = "train_color_10/" if subset in ["train","val"] else "test/"
        dataset_dir = os.path.join(dataset_dir, subset_dir)

        TRAIN_IMAGE_IDS, VAL_IMAGE_IDS = train_test_split(os.listdir(dataset_dir), test_size=0.25, random_state=0)
        # print("-----TRAIN_IMAGE_IDS-------", TRAIN_IMAGE_IDS)
        # print("-----VAL_IMAGE_IDS-------", VAL_IMAGE_IDS)

        # """Train the model."""
        # # Training dataset.
        # dataset_train = KaggleDataset()
        # dataset_train.load_nucleus(dataset_dir, subset)
        # dataset_train.prepare()

        # # Validation dataset
        # dataset_val = KaggleDataset()
        # dataset_val.load_nucleus(dataset_dir, "val")
        # dataset_val.prepare()

        if subset == "val":
            image_ids = VAL_IMAGE_IDS
        else:
            # Get image ids from directory names

            # image_ids = next(os.walk(dataset_dir))[2]
            # if subset == "train":
            #     image_ids = list(set(image_ids) - set(VAL_IMAGE_IDS))
            #     print ()
            image_ids = TRAIN_IMAGE_IDS
        #print("IMAGE ID in load_kaggle", image_ids)
        # Add images
        for image_id in image_ids:
            if image_id != ".DS_Store":
                self.add_image(
                    "kaggle",
                    image_id=image_id,
                    path=os.path.join(dataset_dir, image_id))    ## Change the directory
                # print("========Loading Image (path)========", os.path.join(dataset_dir, "{}".format(image_id)))
                # print("========Loading Image (image_id)========", image_id)


###################### Coco "load_mask" (mutiple class/instances)######################
        """Load instance masks for the given image.
        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].
        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a COCO image, delegate to parent class.
            
    def load_mask(self, image_id):
        
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
        
        image_info = self.image_info[image_id]
        
        # if image_info["source"] != "coco":
        #     return super(CocoDataset, self).load_mask(image_id)

        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        info = self.image_info[image_id]['id']
        info = info[:-4] + '_instanceIds.png'
        
        mask_dir = os.path.join(ROOT_DIR, "train_label_10")
        # print("======== m ========", os.path.join(mask_dir, info))
        # Read mask files from .png image
        
        class_ids = []
        mask = []
        #print('Image Id', image_id)
        
        idlist = [0,33,34,35,36,38,39,40]
        class_dict = {0:0, 33:1, 34:2, 35:3, 36:4, 38:5, 39:6, 40:7}
        
        
        # Read image as array
        #print("-"*50)
        
        m = skimage.io.imread(os.path.join(mask_dir, info))
        class_id = m//1000
        
        #class_bg_sub = np.zeros(m.shape)
        
        for id in idlist:
            if id == 0:
                continue
            else:
#                print("ID!!!!!!", id)
                indices_obj = np.where(class_id==id)
#                print("indices_obj = ", indices_obj)

                if(len(indices_obj[0])>0):
#                    print("entered")
                    class_obj = np.zeros(m.shape)
                    #print("You are here with  id = ", id)
                    class_obj[indices_obj] = 1
                    mask.append(class_obj)
                    class_ids.append(class_dict[id])
                    #class_bg_sub = id*class_obj
    
        #class_bg = np.asarray(class_id - class_bg_sub).astype(np.bool)
        #if np.count_nonzero(class_bg) > 0:
        #   mask.append(class_bg)
        #    class_ids.append(class_dict[0])
        
        #print("mask shape = ", len(mask))
        #print("mask1 shape = ", mask[0].shape)
        #print("class ids shape = ", len(class_ids))
        
        #mask = np.stack(mask, axis=-1)
        if (len(mask)>0):
            mask = np.stack(mask, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
        # print("CLASS IDs", class_ids)
        # print("mask shape=",mask.shape)
        
        return mask, class_ids


##################### Coco "load_mask" (mutiple class/instances)######################

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "kaggle":
            return info["id"]
        else:
            super(self.__class__, self).image_reference(image_id)


############################################################
#  Training
############################################################

def train(model, dataset_dir, subset):
    """Train the model."""
    # Training dataset.
    dataset_train = KaggleDataset()
    dataset_train.load_kaggle(dataset_dir, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = KaggleDataset()
    dataset_val.load_kaggle(dataset_dir, "val")
    dataset_val.prepare()

    # Image augmentation
    # http://imgaug.readthedocs.io/en/latest/source/augmenters.html

    # augmentation = iaa.SomeOf((0, 2), [
    #     iaa.Fliplr(0.5),
    #     iaa.Flipud(0.5),
    #     iaa.OneOf([iaa.Affine(rotate=90),
    #                iaa.Affine(rotate=180),
    #                iaa.Affine(rotate=270)]),
    #     iaa.Multiply((0.8, 1.5)),
    #     iaa.GaussianBlur(sigma=(0.0, 5.0))
    # ])

    # *** This training schedule is an example. Update to your needs ***

    # If starting from imagenet, train heads only for a bit
    # since they have random weights
    print("Train network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=5,
                layers='heads')

    '''
    print("Train all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=40,
                layers='all')
    '''


############################################################
#  RLE Encoding
############################################################

def rle_encode(mask):
    """Encodes a mask in Run Length Encoding (RLE).
    Returns a string of space-separated values.
    """
    assert mask.ndim == 2, "Mask must be of shape [Height, Width]"
    # Flatten it column wise
    m = mask.T.flatten()
    # Compute gradient. Equals 1 or -1 at transition points
    g = np.diff(np.concatenate([[0], m, [0]]), n=1)
    # 1-based indicies of transition points (where gradient != 0)
    rle = np.where(g != 0)[0].reshape([-1, 2]) + 1
    # Convert second index in each pair to lenth
    rle[:, 1] = rle[:, 1] - rle[:, 0]
    return " ".join(map(str, rle.flatten()))


def rle_decode(rle, shape):
    """Decodes an RLE encoded list of space separated
    numbers and returns a binary mask."""
    rle = list(map(int, rle.split()))
    rle = np.array(rle, dtype=np.int32).reshape([-1, 2])
    rle[:, 1] += rle[:, 0]
    rle -= 1
    mask = np.zeros([shape[0] * shape[1]], np.bool)
    for s, e in rle:
        assert 0 <= s < mask.shape[0]
        assert 1 <= e <= mask.shape[0], "shape: {}  s {}  e {}".format(shape, s, e)
        mask[s:e] = 1
    # Reshape and transpose
    mask = mask.reshape([shape[1], shape[0]]).T
    return mask


def mask_to_rle(image_id, mask, scores):
    "Encodes instance masks to submission format."
    assert mask.ndim == 3, "Mask must be [H, W, count]"
    # If mask is empty, return line with image ID only
    if mask.shape[-1] == 0:
        return "{},".format(image_id)
    # Remove mask overlaps
    # Multiply each instance mask by its score order
    # then take the maximum across the last dimension
    order = np.argsort(scores)[::-1] + 1  # 1-based descending
    mask = np.max(mask * np.reshape(order, [1, 1, -1]), -1)
    # Loop over instance masks
    lines = []
    for o in order:
        m = np.where(mask == o, 1, 0)
        # Skip if empty
        if m.sum() == 0.0:
            continue
        rle = rle_encode(m)
        lines.append("{}, {}".format(image_id, rle))
    return "\n".join(lines)


############################################################
#  Detection
############################################################

def detect(model, dataset_dir, subset):
    """Run detection on images in the given directory."""
    print("Running on {}".format(dataset_dir))

    # Create directory
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    submit_dir = "submit_{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())   ## Modify this!!!
    submit_dir = os.path.join(RESULTS_DIR, submit_dir)
    os.makedirs(submit_dir)

    # Read dataset
    dataset = KaggleDataset()
    dataset.load_kaggle(dataset_dir, subset)
    dataset.prepare()
    # Load over images
    submission = []
    for image_id in dataset.image_ids:
        # Load image and run detection
        image = dataset.load_image(image_id)
        # Detect objects
        r = model.detect([image], verbose=0)[0]
        # Encode image to RLE. Returns a string of multiple lines
        source_id = dataset.image_info[image_id]["id"]
        rle = mask_to_rle(source_id, r["masks"], r["scores"])
        submission.append(rle)
        # Save image with masks
        visualize.display_instances(
            image, r['rois'], r['masks'], r['class_ids'],
            dataset.class_names, r['scores'],
            show_bbox=True, show_mask=True,
            title="Predictions")
        plt.savefig("{}/{}".format(submit_dir, dataset.image_info[image_id]["id"]))

    # Save to csv file
    submission = "ImageId,EncodedPixels\n" + "\n".join(submission)
    file_path = os.path.join(submit_dir, "submit.csv")
    with open(file_path, "w") as f:
        f.write(submission)
    print("Saved to ", submit_dir)


############################################################
#  Command Line
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Mask R-CNN for nuclei counting and segmentation')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'detect'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/dataset/",
                        help='Root directory of the dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--subset', required=False,
                        metavar="Dataset sub-directory",
                        help="Subset of dataset to run prediction on")
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "detect":
        assert args.subset, "Provide --subset to run prediction on"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    if args.subset:
        print("Subset: ", args.subset)
    print("Logs: ", args.logs)

    # Configurations
    print("Configurating.......")
    if args.command == "train":
        config = KaggleConfig()
    else:
        config = KaggleInferenceConfig()
    config.display()

    # Create model
    print("Creating Model......")
    if args.command == "train":
        print("Training Mode")
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        print("Inference Mode")
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        print("here")
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()[1]
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("weiightspath", weights_path)
    print("Loading weights ", weights_path)
    print("NOT PAST")
    
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        print("COCO!!!!!!!!!!!!!!")
        
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        print("NOT COCO!!")
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    print("NOT LOADED")
    # Train or evaluate
    if args.command == "train":
        train(model, args.dataset, args.subset)
    elif args.command == "detect":
        detect(model, args.dataset, args.subset)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'detect'".format(args.command))
