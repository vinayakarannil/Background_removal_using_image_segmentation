from __future__ import division

import os
import sys
import tensorflow as tf
import skimage.io as io
import numpy as np
import cv2
import time
import argparse
from PIL import ImageFilter
import multiprocessing
import numpy as np
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

from queue import Queue
from threading import Thread
from app_utils import FPS, WebcamVideoStream

from models.fcn_8s import FCN_8s
from utils.inference import adapt_network_for_any_size_input
from utils.pascal_voc import pascal_segmentation_lut
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
net_shape = (300, 300)
data_format = 'NHWC'

number_of_classes = 21

tf.reset_default_graph()

    
image_filename_placeholder = tf.placeholder(tf.string)
image_tensor = tf.read_file(image_filename_placeholder)

image_tensor = tf.image.decode_jpeg(image_tensor, channels=3)
image_tensor = tf.image.resize_images(image_tensor,(300,250))
# Fake batch for image and annotation by adding
# leading empty axis.
image_batch_tensor = tf.expand_dims(image_tensor, axis=0)

# Be careful: after adaptation, network returns final labels
# and not logits
FCN_8ss = adapt_network_for_any_size_input(FCN_8s, 32)


pred, fcn_16s_variables_mapping = FCN_8ss(image_batch_tensor=image_batch_tensor,
                                          number_of_classes=number_of_classes,
                                          is_training=False)

# The op for initializing the variables.
initializer = tf.local_variables_initializer()

saver = tf.train.Saver()

sess = tf.Session() 
sess.run(initializer)
saver.restore(sess, tf.train.latest_checkpoint('ckpt'))
print("restored")
def segment_img(image_filename, sess):

    feed_dict_to_use = {image_filename_placeholder: image_filename}

    image_np, pred_np = sess.run([image_tensor, pred], feed_dict=feed_dict_to_use)
       

    import skimage.morphology
         
    prediction_mask = (pred_np.squeeze() == 15)
    background_mask = (pred_np.squeeze() != 15)

    # Let's apply some morphological operations to
    # create the contour for our sticker

    cropped_object = image_np * np.dstack((prediction_mask,) * 3)
    

    cropped_background= image_np * np.dstack((background_mask,) * 3)


    square = skimage.morphology.square(5)

    temp = skimage.morphology.binary_erosion(prediction_mask, square)
    
    temp2 = skimage.morphology.binary_erosion(background_mask, square)


    negative_mask = (temp != True)

    negative_mask2 = (temp2 != True)


    png_transparancy_mask = np.uint8(prediction_mask * 255)

    png_transparancy_mask2 = np.uint8(background_mask * 255)    
    
    image_shape = cropped_object.shape
    bg_shape = cropped_background.shape
    
    png_array = np.zeros(shape=[image_shape[0], image_shape[1], 4], dtype=np.uint8)
    
    png_array2 = np.zeros(shape=[bg_shape[0], bg_shape[1], 4], dtype=np.uint8)
    print(png_array2.shape)
    png_array[:, :, :3] = cropped_object
    png_array2[:, :, :3] = cropped_background
    png_array[:, :, 3] = png_transparancy_mask
    png_array2[:, :, 3] = png_transparancy_mask2
  

    image_array = Image.fromarray(png_array)
    bg_array = Image.fromarray(png_array2)

    img = image_array.convert("RGBA")


    bg_ = bg_array.convert("RGBA")

    datas = img.getdata()

    newData = []
    for item in datas:
        if item[3] == 0:

           newData.append((255,255,255,255))
        else:
           newData.append(item)

    img.putdata(newData)
    img.save("person.png", "png")  

    bg_.save("background.png", "png")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-src', '--source', dest='video_source', type=int,
                        default=0, help='Device index of the camera.')
    parser.add_argument('-wd', '--width', dest='width', type=int,
                        default=480, help='Width of the frames in the video stream.')
    parser.add_argument('-ht', '--height', dest='height', type=int,
                        default=360, help='Height of the frames in the video stream.')
    args = parser.parse_args()
    segment_img("ppp.png", sess)
    
