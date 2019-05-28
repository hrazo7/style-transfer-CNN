from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os
import time
import functools
import numpy as np
import IPython.display as display
import matplotlib.pyplot as plt
import matplotlib as mpl
import tensorflow as tf


mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False

content_pic = os.getcwd() + 'content_pics/me.jpg'
style_pic = os.getcwd() + 'style_pics/obama.jpeg'

#load the image
def load_image(image_path):
  max_dim = 512
  img = tf.io.read_file(image_path)
  img = tf.image.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)

  shape = tf.cast(tf.shape(img)[:-1], tf.float32)
  long_dim = max(shape)
  scale = max_dim / long_dim

  new_shape = tf.cast(shape * scale, tf.int32)

  img = tf.image.resize(img, new_shape)
  img = img[tf.newaxis, :]
  return img


#show the image
def show_image(image, title):

    if len(image.shape) > 3:
        image = tf.squeeze(image, axis-0)

    plt.imshow(image, title)


content_image = load_image(content_pic)
style_image = load_image(style_pic)

plt.subplot(1, 2, 1)
show_image(content_image, "Content Image")

plt.subplot(1, 2, 2)
show_image(style_image, "Style Image")




