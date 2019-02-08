from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.models as models
import copy
from PIL import Image

content_path = ('/Users/hernanrazo/pythonProjects/style_transfer_CNN/content_pics/me.jpg')
style_path = ('/Users/hernanrazo/pythonProjects/style_transfer_CNN/style_pics/obama.jpeg')
result_path = ('/Users/hernanrazo/pythonProjects/style_transfer_CNN/result_pics/result.jpeg')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
img_size = 512 if torch.cuda.is_available() else 128 
loader = transforms.Compose([transforms.Resize(img_size), transforms.ToTensor()])

def load_image(path):

	image = Image.open(path)
	image = loader(image).unsqueeze(0)

	return image.to(device, torch.float)

def show_image(tensor, title = None):

	image = tensor.cpu().clone()
	image = image.squeeze(0)
	image = unloader(image)

	if title is not None:

		plt.title(title)
	plt.pause(0.001)


#---------------------------------------------

content_image = load_image(content_path)
style_image = load_image(style_path)



unloader = transforms.ToPILImage()
plt.ion()

plt.figure
show_image(style_image, title='Style Image')

plt.figure
show_image(content_image, title='content Image')


























'''
import numpy as np
import time
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from tensorflow.python.keras import models 
from tensorflow.python.keras import losses
from tensorflow.python.keras import layers
from tensorflow.python.keras.preprocessing import image as kp_image

from PIL import Image
from keras import backend as k
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.layers import Input
from scipy.optimize import fmin_l_bfgs_b

content_path = ('/Users/hernanrazo/pythonProjects/style_transfer_CNN/content_pics/me.jpg')
style_path = ('/Users/hernanrazo/pythonProjects/style_transfer_CNN/style_pics/obama.jpeg')
result_path = ('/Users/hernanrazo/pythonProjects/style_transfer_CNN/result_pics/result.jpeg')


def load_img(path):

	max_dim = 512
	img = Image.open(path)
	long = max(img.size)
	scale = max_dim / long
	img = img.resize((round(img.size[0]*scale), round(img.size[1]*scale)), Image.ANTIALIAS)
	img = kp_image.img_to_array(img)
	img = np.expand_dims(img, axis=0)
	
	return img

def load_and_process_img(path):
	img = load_img(path)
	img = tf.keras.applications.vgg19.preprocess_input(img)
	
	return img

def deprocess_img(processed_img):

	x = processed_img.copy()
	if len(x.shape) == 4:
		
		x = np.squeeze(x, O)
	assert len(x.shape) == 3, ('what do I put here?')

	if len(x.shape) != 3:

		raise valueError("Invalid input")

	x[:, :, 0] += 103.939
	x[:, :, 1] += 116.779
	x[:, : 2] += 123.68
	x = x[:, :, ::-1]

	x = np.clip(x, 0, 255).astype('uint8')

	return x

content_layers = ['block4_conv2']
style_layers = ['block1_conv1', 
'block2_conv1', 
'block3_conv1', 
'block4_conv1', 
'block5_conv1']

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

def get_model():

	vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
	vgg.trainable = False	
	style_outputs = [vgg.get_layer(name).output for name in style_layers]
	content_outputs = [vgg.get_layer(name).output for name in content_layers]
	model_outputs = style_outputs + content_outputs

	return models.Model(vgg.input, model_outputs)

def content_loss(base_content, target):

	content_loss = tf.reduce_mean(tf.square(base_content - target))
	
	return content_loss

def gram_matrix(input_tensor):

	channels = int(input_tensor.shape[-1])
	a = tf.reshape(input_tensor, [-1, channels])
	n = tf.shape(a)[0]
	gram = tf.matmul(a, a, transpose_a = True)
	result = gram / tf.cast(n, tf.float32)
	
	return result

def style_loss(base_style, gram_target):

	height = base_style.get_shape().as_list()
	width = base_style.get_shape().as_list()
	channels = base_style.get_shape().as_list()
	gram_style = gram_matrix(base_style)
	result = tf.reduce_mean(tf.square(gram_style - gram_target))

	return result

def feature_rep(model, content_path, style_path):

	content_image = load_and_process_img(content_path)
	style_image = load_and_process_img(style_path)

	style_outputs = model(style_image)
	content_outputs = model(content_image)

	style_features = [style_layer[0] for style_layer in style_outputs[:num_style_layers]]
	content_features = [content_layer[0] for content_layer in content_outputs[num_style_layers:]]

	return style_features
	return content_features

def  calculate_loss (model, loss_weights, init_image, gram_style_features, content_features):

	style_weight = loss_weights
	content_weight = loss_weights
	total_variation_weight = loss_weights

	model_outputs = model(init_image)

	style_output_features = model_outputs[:num_style_layers]
	content_output_features = model_outputs[num_style_layers:]

	style_score = 0
	content_score = 0

	weight_per_style_layer = 1.0 / float(num_style_layers)

	for target_content, comb_content in zip(content_features, content_output_features):
		style_score += weight_per_style_layer * style_loss(comb_content[0], target_style)
	
	weight_per_content_layer = 1.0 / float(num_content_layers)

	for target_content, comb_content in zip(content_features, content_output_features):
		content_score += weight_per_content_layer * content_loss(comb_content[0], target_content)

	style_score += style_weight
	content_score += content_weight
	total_variation_score = total_variation_weight * total_variation_loss(init_image)

	total_loss = style_score + content_score + total_variation_score

	return total_loss
	return style_score
	return content_score
	return total_variation_score

def calculate_gradients(cfg):

	with tf.GradientTape() as tape:

		all_loss = calculate_loss(**cfg)

	total_loss = all_loss[0]

	return tape.gradient(total_loss, cfg['init_image'])
	return all_loss

def run_style_transfer(content_path, 
	style_path, 
	num_iteration=1000, 
	content_weight=1e3, 
	style_weight=1e-2):

	display_num = 100
	model = get_model()

	for layer in model.layers:
		layer.trainable = False

	style_features = feature_rep(model, content_path, style_path)
	content_features = feature_rep(model, content_path, style_path)
	gram_style_features = [gram_matrix(style_features) for style_feature in style_features]

	init_image = load_and_process_img(content_path)
	init_image = tfe.Variable(init_image, dtype=tf.float32)

	optimizer = tf.train.AdamOptimizer(learning_rate=10.0)

	iter_count = 1

	best_loss = float('inf'), None
	best_img = float('inf'), None

	print('Total time: {:.4f}s'.format(time.time() - global_start))

	best_img = img.save(result_path)
	return best_img
	return best_loss

best = run_style_transfer(content_path, style_path)
best_loss = run_style_transfer(content_path, style_path)

'''
