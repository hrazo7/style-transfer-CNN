import numpy as np
import time
from PIL import Image
from keras import backend as k
from keras.preprocessing.image import load_img, img_to_array
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input
from scipy.optimize import fmin_l_bfgs_b

#set paths for images 
source_path = ('/Users/hernanrazo/pythonProjects/style_transfer_CNN/source_pics/me.jpg')
style_path = ('/Users/hernanrazo/pythonProjects/style_transfer_CNN/style_pics/obama.jpeg')
result_path = ('/Users/hernanrazo/pythonProjects/style_transfer_CNN/result_pics/result.jpeg')

#et target size for all images
target_size = (512, 512)

#open and process images
source_image = Image.open(source_path)
source_image_size = source_image.size 
load_source_image = load_img(path = source_path, target_size = target_size)
source_image_array = img_to_array(source_image)
source_image_array = k.variable(preprocess_input(np.expand_dims(source_image_array, axis = 0)), dtype = 'float32')

style_image = load_img(path = style_path, target_size = target_size)
style_image_array = img_to_array(style_image)
style_image_array = k.variable(preprocess_input(np.expand_dims(style_image, axis = 0)), dtype = 'float32')

result_image = np.random.randint(256, size = (512, 512, 3)).astype('float64')
result_image = preprocess_input(np.expand_dims(result_image, axis=0))
result_image_placeholder = k.placeholder(shape=(1, 512, 512, 3))











