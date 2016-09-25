# Imports
import numpy as np
import tensorflow as tf
import sys
import os
import matplotlib.pyplot as plt
from skimage.transform import resize

from libs import utils

# Init
plt.style.use('ggplot')

# Gather the image files
dirname = './img_align_celeba'

filenames = [os.path.join(dirname, fname) for fname in os.listdir(dirname)]

filenames = filenames[:100]
assert(len(filenames) == 100)

# Read the files as images
imgs = [plt.imread(fname)[:,:,:3] for fname in filenames]

imgs = [resize(img_i, (100, 100)) for img_i in imgs]

imgs = np.array(imgs).astype(np.float32)

assert(imgs.shape == (100, 100, 100, 3))
plt.figure(figsize=(10,10))
plt.imshow(utils.montage(imgs, saveto='dataset.png'))

##########
sess = tf.Session()

mean_img_op = tf.reduce_mean(imgs, reduction_indices=0, name='mean')

mean_img = sess.run(mean_img_op)

assert(mean_img.shape == (100, 100, 3))
plt.figure(figsize=(10, 10))
plt.imshow(mean_img)
plt.imsave(arr=mean_img, fname='mean.png')

###
mean_img_4d = tf.reshape(mean_img, [1, 100, 100, 3])

subtraction = imgs - mean_img_4d

std_img_op = tf.sqrt(tf.reduce_sum(subtraction * subtraction, 
reduction_indices=0))

std_img = sess.run(std_img_op)

assert(std_img.shape == (100, 100) or std_img.shape == (100, 100, 3))
plt.figure(figsize=(10, 10))
std_img_show = std_img / np.max(std_img)
plt.imshow(std_img_show)
plt.imsave(arr=std_img_show, fname='std.png')

##
norm_imgs_op = tf.div(tf.sub(imgs, mean_img_4d), std_img)

norm_imgs = sess.run(norm_imgs_op)
print(np.min(norm_imgs), np.max(norm_imgs))
print(imgs.dtype)

assert(norm_imgs.shape == (100, 100, 100, 3))
plt.figure(figsize=(10, 10))
plt.imshow(utils.montage(norm_imgs, 'normalized_bad.png'))

norm_imgs_show = (norm_imgs - np.min(norm_imgs)) / (np.max(norm_imgs) - np.min(norm_imgs))
plt.figure(figsize=(10, 10))
plt.imshow(utils.montage(norm_imgs_show, 'normalized.png'))