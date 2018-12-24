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

def feature_representations(result, layer_names, model):

	feat_matrices = []

	for layer_name in layer_names:

		selected_layer = model.get_layer(layer_name)
		raw_feature = selected_layer.output
		raw_feature_shape = k.shape(raw_feature).eval(session=tf_session)

		num_filters = raw_feature_shape[-1]
		num_s_elements = raw_feature_shape[1]*raw_feature_shape[2]

		feature_matrix = k.reshape(raw_feature, (num_s_elements, num_filters))
		feature_matrix = k.transpose(feature_matrix)
		feature_matrices.append(feature_matrix)

	return feat_matrices


def gram_matrix(result_features):

	gram_matrix = k.dot(result_features, k.transpose(result_features))
	return gram_matrix


def source_loss_fun(result_features, source_features):

	source_loss_fun = 0.5*k.sum(k.square(result_features - source_features))
	return source_loss_fun


def style_loss_fun(ws, Gs, As):

	style_loss = k.variable(0.)

	for w, G, A in zip(ws, Gs, As):

		num_s_elements = k.int_shape(G)[1]
		raw_feature_shape = k.int_shape(G)[0]
		
		result_g_matrix = gram_matrix(G)
		style_g_matrix = gram_matrix(A)

		style_loss += w*0.25*k.sum(k.square(result_g_matrix - style_g_matrix))/
		(raw_feature_shape**2 * num_s_elements**2)
	
	return style_loss


def total_loss(result_image_placeholder, alpha = 1.0, beta = 10000.0):

	result_features = feature_representations(result_image_placeholder, layer_names=[source_layer_name], model=result_model)[0]
	Gs = feature_representations(result_image_placeholder, layer_names=source_layer_names, )
















