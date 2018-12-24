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

#set target size for all images
target_height = 512
target_width = 512
target_size = (target_height, target_width)

source_image = Image.open(source_path)
source_image_size = source_image.size 

load_source_image = load_img(path = source_path, target_size = target_size)
source_image_array = img_to_array(source_image)
source_image_array = k.variable(preprocess_input(np.expand_dims(source_image_array, axis = 0)), dtype = 'float32')

load_style_image = load_img(path = style_path, target_size = target_size)
style_image_array = img_to_array(load_style_image)
style_image_array = k.variable(preprocess_input(np.expand_dims(load_style_image, axis = 0)), dtype = 'float32')

result_image = np.random.randint(256, size = (512, 512, 3)).astype('float64')
result_image = preprocess_input(np.expand_dims(result_image, axis=0))
result_image_placeholder = k.placeholder(shape=(1, 512, 512, 3))

def feature_representations(result, layer_names, model):

	feature_matrices = []

	for layer_name in layer_names:

		selected_layer = model.get_layer(layer_name)
		raw_feature = selected_layer.output
		raw_feature_shape = k.shape(raw_feature).eval(session=tf_session)

		num_filters = raw_feature_shape[-1]
		num_s_elements = raw_feature_shape[1]*raw_feature_shape[2]

		feature_matrix = k.reshape(raw_feature, (num_s_elements, num_filters))
		feature_matrix = k.transpose(feature_matrix)
		feature_matrices.append(feature_matrix)

	return feature_matrices


def gram_matrix(result_features):

	gram_matrix = k.dot(result_features, k.transpose(result_features))
	return gram_matrix


def calc_source_loss(result_features, source_features):

	source_loss_fun = 0.5*k.sum(k.square(result_features - source_features))
	return source_loss_fun


def calc_style_loss(ws, Gs, As):

	style_loss = k.variable(0.)

	for w, G, A in zip(ws, Gs, As):

		num_s_elements = k.int_shape(G)[1]
		raw_feature_shape = k.int_shape(G)[0]
		
		result_g_matrix = gram_matrix(G)
		style_g_matrix = gram_matrix(A)

		style_loss += w*0.25*k.sum(k.square(result_g_matrix - style_g_matrix))/(raw_feature_shape**2 * num_s_elements**2)
	
	return style_loss


def total_loss(result_image_placeholder, alpha = 1.0, beta = 10000.0):

	result_features = feature_representations(result_image_placeholder, layer_names=[source_layer_name], model=result_model)[0]
	Gs = feature_representations(result_image_placeholder, layer_names=style_layer_names, model=result_model)

	source_loss = calc_source_loss(result_features, source_features)
	style_loss = calc_style_loss(ws, Gs, As)
	total_loss = alpha*source_loss + beta*style_loss

	return total_loss


def calculate_loss(result_array):

	if result_array.shape != (1, 512, 512, 3):

		result_array = result_array.reshape((1, 512, 512, 3))

	loss_fun = k.function([result_model.input], k.gradients(total_loss(result_model.input), [result_model.input]))
	
	return loss_fun([result_array])[0].astype('float64')


def get_gradient(result_array):

	if result_array.shape != (1, 512, 512, 3):

		result_array = result_array.reshape((1, 512, 512, 3))

	gradient_fun = k.function([result_model.input], k.gradients(total_loss(result_model.input), [result_model.input]))
	gradient = gradient_fun([result_array])[0].flatten().astype('float64')

	return gradient

def postprocess_array(x):

	if x.shape != (512, 512, 3):

		x = x.reshape((512, 512, 3))

	x[..., 0] += 103.939
	x[..., 1] += 116.779
	x[..., 2] += 123.68

	x = x[..., ::-1]
	x = np.clip(x, 0, 255)
	x = x.astype('uint8')

	return x 

def reproccess_array(x):

	x = x.np.expand_dims(x.astype('float64'), axis=0)
	x = preprocess_input(x)

	return x

def save_original_size(x, target_size=source_image_size):

	xIm = Image.fromarray(x)
	xIm = xIm.resize(target_size)
	xIm.save(result_path)

	return xIm

#actually start session
tf_session = k.get_session()
source_model = VGG16(include_top=False, weights='imagenet', input_tensor=source_image_array)
style_model = VGG16(include_top=False, weights='imagenet', input_tensor=style_image_array)
result_model = VGG16(include_top=False, weights='imagenet', input_tensor=result_image_placeholder)

source_layer_name = 'block4_conv2'
style_layer_names = ['block1_conv1', 'block2_conv1', 'block3_conv1','block4_conv1']

source_features= feature_representations(result=source_image_array, layer_names=[source_layer_name], model=source_model)[0]
As = feature_representations(result=style_image_array, layer_names=style_layer_names, model=style_model)
ws = np.ones(len(style_layer_names))/float(len(style_layer_names))

iterations = 600
x_value = result_image.flatten()
xopt, resultfeatures_value, info = fmin_l_bfgs_b(calculate_loss, x_value, fprime=get_gradient, maxiter=iterations, disp=True)

x_out = postprocess.array(xopt)
xIm = save_original_size(x_out)

print('image saved')
end = time.time()

print('Time take: {}'.format(end-start))

