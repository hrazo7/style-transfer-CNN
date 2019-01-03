
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

	channels = np.array(input_tensor.shape[-1])
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
'''













