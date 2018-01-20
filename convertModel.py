
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

CAFFE_ROOT = '/home/manish/caffe-rc5/'
import sys
sys.path.insert(0, CAFFE_ROOT + 'python')
import caffe
import glob
from PIL import Image
import numpy as np


AVA_ROOT = '/home/manish/projects/deepImageAestheticsAnalysis/AVA_Model/'
INIT_ROOT = '/home/manish/projects/deepImageAestheticsAnalysis/initModel/'
IMAGE_MEAN= AVA_ROOT + 'imagenet_mean.binaryproto'
DEPLOY = INIT_ROOT  + 'initModel.prototxt'
MODEL_FILE = INIT_ROOT + 'initModel.caffemodel'

IMAGE_WIDTH = 227
IMAGE_HEIGHT = 227
dims = 227,227

caffe.set_mode_cpu()
net = caffe.Net(DEPLOY, MODEL_FILE, caffe.TEST)

#############################Image Data##################
files = glob.glob('./Data/*.jpeg')
images = np.zeros((len(files),IMAGE_WIDTH,IMAGE_HEIGHT,3))
for i,f in enumerate(files):
    img = Image.open(f)
    img = img.resize(dims,Image.ANTIALIAS)
    img = np.array(img)
    img[:, :, 0], img[:, :, 2] = img[:, :, 2], img[:, :, 0]
    images[i,:,:,:] = img

############# TensorFlow Model ####################

import tensorflow as tf

def print_activations(t):
  print(t.op.name, ' ', t.get_shape().as_list())

def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w, padding="VALID", group=1):

    c_i = input.get_shape()[-1]
    assert c_i % group == 0
    assert c_o % group == 0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)

    if group == 1:
        conv = convolve(input, kernel)
    else:
        group_sz = int((input.get_shape().as_list())[-1] / group)
        input_sz = [group_sz] * group
        input_groups = tf.split(input, input_sz, 3)

        group_sz = int((kernel.get_shape().as_list())[-1] / group)
        kernel_sz = [group_sz] * group
        kernel_groups = tf.split(kernel, kernel_sz, 3)
        output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
        conv = tf.concat(output_groups,3)
    return tf.reshape(tf.nn.bias_add(conv, biases), [-1] + conv.get_shape().as_list()[1:])


x = tf.placeholder(tf.float32,(None,IMAGE_WIDTH,IMAGE_HEIGHT,3))


######################### conv1 ###########################
with tf.name_scope('conv1') as scope:
    net_weights = net.params['conv1'][0].data
    net_weights = np.transpose(net_weights,(2,3,1,0))
    kernel = tf.Variable(net_weights,name="weights")
    biases = tf.Variable(net.params['conv1'][1].data,name="bias")
    conv_in = conv(x,kernel,biases,11,11,96,4,4, padding='SAME')
    conv1 = tf.nn.relu(conv_in)

with tf.name_scope('lrn1') as scope:
    lrn1 = tf.nn.local_response_normalization(conv1,
                                              alpha=1e-4,
                                              beta=0.75,
                                              depth_radius=2,
                                              bias=2.0)

# pool1
pool1 = tf.nn.max_pool(lrn1,
                     ksize=[1, 3, 3, 1],
                     strides=[1, 2, 2, 1],
                     padding='VALID',
                     name='pool1')
print_activations(pool1)


################################# conv2#####################################
with tf.name_scope('conv2') as scope:
    net_weights = net.params['conv2'][0].data
    net_weights = np.transpose(net_weights,(2,3,1,0))
    kernel = tf.Variable(net_weights,name="weights")
    biases = tf.Variable(net.params['conv2'][1].data,name="bias")
    conv_in = conv(pool1,kernel,biases,5,5,256,1,1, padding='SAME',group=2)
    conv2 = tf.nn.relu(conv_in)

with tf.name_scope('lrn2') as scope:
    lrn2 = tf.nn.local_response_normalization(conv2,
                                              alpha=1e-4,
                                              beta=0.75,
                                              depth_radius=2,
                                              bias=2.0)

# pool1
pool2 = tf.nn.max_pool(lrn2,
                     ksize=[1, 3, 3, 1],
                     strides=[1, 2, 2, 1],
                     padding='VALID',
                     name='pool2')
print_activations(pool2)


################################# conv3#####################################
with tf.name_scope('conv3') as scope:
    net_weights = net.params['conv3'][0].data
    net_weights = np.transpose(net_weights,(2,3,1,0))
    kernel = tf.Variable(net_weights,name="weights")
    biases = tf.Variable(net.params['conv3'][1].data,name="bias")
    conv_in = conv(pool2,kernel,biases,3,3,384,1,1, padding='SAME',group=1)
    conv3 = tf.nn.relu(conv_in)
    print_activations(conv3)


################################# conv4#####################################
with tf.name_scope('conv4') as scope:
    net_weights = net.params['conv4'][0].data
    net_weights = np.transpose(net_weights,(2,3,1,0))
    kernel = tf.Variable(net_weights,name="weights")
    biases = tf.Variable(net.params['conv4'][1].data,name="bias")
    conv_in = conv(conv3,kernel,biases,3,3,384,1,1, padding='SAME',group=2)
    conv4 = tf.nn.relu(conv_in)
    print_activations(conv4)


################################# conv5#####################################
with tf.name_scope('conv5') as scope:
    net_weights = net.params['conv5'][0].data
    net_weights = np.transpose(net_weights,(2,3,1,0))
    kernel = tf.Variable(net_weights,name="weights")
    biases = tf.Variable(net.params['conv5'][1].data,name="bias")
    conv_in = conv(conv4,kernel,biases,3,3,256,1,1, padding='SAME',group=2)
    conv5 = tf.nn.relu(conv_in)
    print_activations(conv5)

pool5 = tf.nn.max_pool(conv5,
                       ksize=[1, 3, 3, 1],
                       strides=[1, 2, 2, 1],
                       padding='VALID',
                       name='pool5')
print_activations(pool5)

################################# fc6#####################################
with tf.name_scope('fc6') as scope:
    flatten = tf.reshape(pool5, [-1, int(np.prod(pool5.get_shape()[1:]))])
    net_weights = net.params['fc6'][0].data
    net_weights = np.transpose(net_weights,(1,0))
    fc6_weights = tf.Variable(net_weights,name="weights")
    biases = tf.Variable(net.params['fc6'][1].data,name="bias")
    fullWeights = tf.nn.bias_add(tf.matmul(flatten,net_weights),biases)
    fc6 = tf.nn.relu(fullWeights)
    print_activations(fc6)


################################# fc7#####################################
with tf.name_scope('fc7') as scope:
    net_weights = net.params['fc7'][0].data
    net_weights = np.transpose(net_weights,(1,0))
    fc7_weights = tf.Variable(net_weights,name="weights")
    biases = tf.Variable(net.params['fc7'][1].data,name="bias")
    fullWeights = tf.nn.bias_add(tf.matmul(fc6,net_weights),biases)
    fc7 = tf.nn.relu(fullWeights)
    print_activations(fc7)

################################# fc8new#####################################
with tf.name_scope('fc8new') as scope:
    net_weights = net.params['fc8new'][0].data
    net_weights = np.transpose(net_weights,(1,0))
    fc8new_weights = tf.Variable(net_weights,name="weights")
    biases = tf.Variable(net.params['fc8new'][1].data,name="bias")
    fullWeights = tf.nn.bias_add(tf.matmul(fc7,net_weights),biases)
    fc8new = tf.nn.relu(fullWeights)


################################# fc8 Balancing Elements#####################################
with tf.name_scope('fc8_BalancingElement') as scope:
    net_weights = net.params['fc8_BalancingElement'][0].data
    net_weights = np.transpose(net_weights,(1,0))
    fc8_BalancingElement_weights = tf.Variable(net_weights,name="weights")
    biases = tf.Variable(net.params['fc8_BalancingElement'][1].data,name="bias")
    fullWeights = tf.nn.bias_add(tf.matmul(fc7,net_weights),biases)
    fc8_BalancingElement = tf.nn.relu(fullWeights)

    net_weights = net.params['fc9_BalancingElement'][0].data
    net_weights = np.transpose(net_weights,(1,0))
    fc9_BalancingElement_weights = tf.Variable(net_weights,name="weights")
    biases = tf.Variable(net.params['fc9_BalancingElement'][1].data,name="bias")
    fullWeights = tf.nn.bias_add(tf.matmul(fc8_BalancingElement,net_weights),biases)
    fc9_BalancingElement = tf.nn.relu(fullWeights)


################################# fc8 Color Harmony#####################################
with tf.name_scope('fc8_ColorHarmony') as scope:
    net_weights = net.params['fc8_ColorHarmony'][0].data
    net_weights = np.transpose(net_weights,(1,0))
    fc8_ColorHarmony_weights = tf.Variable(net_weights,name="weights")
    biases = tf.Variable(net.params['fc8_ColorHarmony'][1].data,name="bias")
    fullWeights = tf.nn.bias_add(tf.matmul(fc7,net_weights),biases)
    fc8_ColorHarmony = tf.nn.relu(fullWeights)

    net_weights = net.params['fc9_ColorHarmony'][0].data
    net_weights = np.transpose(net_weights,(1,0))
    fc9_ColorHarmony_weights = tf.Variable(net_weights,name="weights")
    biases = tf.Variable(net.params['fc9_ColorHarmony'][1].data,name="bias")
    fullWeights = tf.nn.bias_add(tf.matmul(fc8_ColorHarmony,net_weights),biases)
    fc9_ColorHarmony = tf.nn.relu(fullWeights)

################################# fc8 Content #####################################
with tf.name_scope('fc8_Content') as scope:
    net_weights = net.params['fc8_Content'][0].data
    net_weights = np.transpose(net_weights,(1,0))
    fc8_Content_weights = tf.Variable(net_weights,name="weights")
    biases = tf.Variable(net.params['fc8_Content'][1].data,name="bias")
    fullWeights = tf.nn.bias_add(tf.matmul(fc7,net_weights),biases)
    fc8_Content = tf.nn.relu(fullWeights,name=scope)

    net_weights = net.params['fc9_Content'][0].data
    net_weights = np.transpose(net_weights,(1,0))
    fc9_Content_weights = tf.Variable(net_weights,name="weights")
    biases = tf.Variable(net.params['fc9_Content'][1].data,name="bias")
    fullWeights = tf.nn.bias_add(tf.matmul(fc8_Content,net_weights),biases)
    fc9_Content = tf.nn.relu(fullWeights)

################################# fc8 DoF #####################################
with tf.name_scope('fc8_DoF') as scope:
    net_weights = net.params['fc8_DoF'][0].data
    net_weights = np.transpose(net_weights,(1,0))
    fc8_DoF_weights = tf.Variable(net_weights,name="weights")
    biases = tf.Variable(net.params['fc8_DoF'][1].data,name="bias")
    fullWeights = tf.nn.bias_add(tf.matmul(fc7,net_weights),biases)
    fc8_DoF = tf.nn.relu(fullWeights)

    net_weights = net.params['fc9_DoF'][0].data
    net_weights = np.transpose(net_weights,(1,0))
    fc9_DoF_weights = tf.Variable(net_weights,name="weights")
    biases = tf.Variable(net.params['fc9_DoF'][1].data,name="bias")
    fullWeights = tf.nn.bias_add(tf.matmul(fc8_DoF,net_weights),biases)
    fc9_DoF = tf.nn.relu(fullWeights)


################################# fc8 Light #####################################
with tf.name_scope('fc8_Light') as scope:
    net_weights = net.params['fc8_Light'][0].data
    net_weights = np.transpose(net_weights,(1,0))
    fc8_Light_weights = tf.Variable(net_weights,name="weights")
    biases = tf.Variable(net.params['fc8_Light'][1].data,name="bias")
    fullWeights = tf.nn.bias_add(tf.matmul(fc7,net_weights),biases)
    fc8_Light = tf.nn.relu(fullWeights)

    net_weights = net.params['fc9_Light'][0].data
    net_weights = np.transpose(net_weights,(1,0))
    fc9_Light_weights = tf.Variable(net_weights,name="weights")
    biases = tf.Variable(net.params['fc9_Light'][1].data,name="bias")
    fullWeights = tf.nn.bias_add(tf.matmul(fc8_Light,net_weights),biases)
    fc9_Light = tf.nn.relu(fullWeights)

################################# fc8 MotionBlur #####################################
with tf.name_scope('fc8_MotionBlur') as scope:
    net_weights = net.params['fc8_MotionBlur'][0].data
    net_weights = np.transpose(net_weights,(1,0))
    fc8_MotionBlur_weights = tf.Variable(net_weights,name="weights")
    biases = tf.Variable(net.params['fc8_MotionBlur'][1].data,name="bias")
    fullWeights = tf.nn.bias_add(tf.matmul(fc7,net_weights),biases)
    fc8_MotionBlur = tf.nn.relu(fullWeights)

    net_weights = net.params['fc9_MotionBlur'][0].data
    net_weights = np.transpose(net_weights,(1,0))
    fc9_MotionBlur_weights = tf.Variable(net_weights,name="weights")
    biases = tf.Variable(net.params['fc9_MotionBlur'][1].data,name="bias")
    fullWeights = tf.nn.bias_add(tf.matmul(fc8_MotionBlur,net_weights),biases)
    fc9_MotionBlur = tf.nn.relu(fullWeights)


################################# fc8 Object #####################################
with tf.name_scope('fc8_Object') as scope:
    net_weights = net.params['fc8_Object'][0].data
    net_weights = np.transpose(net_weights,(1,0))
    fc8_Object_weights = tf.Variable(net_weights,name="weights")
    biases = tf.Variable(net.params['fc8_Object'][1].data,name="bias")
    fullWeights = tf.nn.bias_add(tf.matmul(fc7,net_weights),biases)
    fc8_Object = tf.nn.relu(fullWeights)

    net_weights = net.params['fc9_Object'][0].data
    net_weights = np.transpose(net_weights,(1,0))
    fc9_Object_weights = tf.Variable(net_weights,name="weights")
    biases = tf.Variable(net.params['fc9_Object'][1].data,name="bias")
    fullWeights = tf.nn.bias_add(tf.matmul(fc8_Object,net_weights),biases)
    fc9_Object = tf.nn.relu(fullWeights)

################################# fc8 Repetition #####################################
with tf.name_scope('fc8_Repetition') as scope:
    net_weights = net.params['fc8_Repetition'][0].data
    net_weights = np.transpose(net_weights,(1,0))
    fc8_Repetition_weights = tf.Variable(net_weights,name="weights")
    biases = tf.Variable(net.params['fc8_Repetition'][1].data,name="bias")
    fullWeights = tf.nn.bias_add(tf.matmul(fc7,net_weights),biases)
    fc8_Repetition = tf.nn.relu(fullWeights)

    net_weights = net.params['fc9_Repetition'][0].data
    net_weights = np.transpose(net_weights,(1,0))
    fc9_Repetition_weights = tf.Variable(net_weights,name="weights")
    biases = tf.Variable(net.params['fc9_Repetition'][1].data,name="bias")
    fullWeights = tf.nn.bias_add(tf.matmul(fc8_Repetition,net_weights),biases)
    fc9_Repetition = tf.nn.relu(fullWeights)

################################# fc8 RuleOfThirds #####################################
with tf.name_scope('fc8_RuleOfThirds') as scope:
    net_weights = net.params['fc8_RuleOfThirds'][0].data
    net_weights = np.transpose(net_weights,(1,0))
    fc8_RuleOfThirds_weights = tf.Variable(net_weights,name="weights")
    biases = tf.Variable(net.params['fc8_RuleOfThirds'][1].data,name="bias")
    fullWeights = tf.nn.bias_add(tf.matmul(fc7,net_weights),biases)
    fc8_RuleOfThirds = tf.nn.relu(fullWeights)

    net_weights = net.params['fc9_RuleOfThirds'][0].data
    net_weights = np.transpose(net_weights,(1,0))
    fc9_RuleOfThirds_weights = tf.Variable(net_weights,name="weights")
    biases = tf.Variable(net.params['fc9_RuleOfThirds'][1].data,name="bias")
    fullWeights = tf.nn.bias_add(tf.matmul(fc8_RuleOfThirds,net_weights),biases)
    fc9_RuleOfThirds = tf.nn.relu(fullWeights)

################################# fc8 Symmetry #####################################
with tf.name_scope('fc8_Symmetry') as scope:
    net_weights = net.params['fc8_Symmetry'][0].data
    net_weights = np.transpose(net_weights, (1, 0))
    fc8_Symmetry_weights = tf.Variable(net_weights, name="weights")
    biases = tf.Variable(net.params['fc8_Symmetry'][1].data, name="bias")
    fullWeights = tf.nn.bias_add(tf.matmul(fc7, net_weights), biases)
    fc8_Symmetry = tf.nn.relu(fullWeights)

    net_weights = net.params['fc9_Symmetry'][0].data
    net_weights = np.transpose(net_weights, (1, 0))
    fc9_Symmetry_weights = tf.Variable(net_weights, name="weights")
    biases = tf.Variable(net.params['fc9_Symmetry'][1].data, name="bias")
    fullWeights = tf.nn.bias_add(tf.matmul(fc8_Symmetry, net_weights), biases)
    fc9_Symmetry = tf.nn.relu(fullWeights)

################################# fc8 VividColor #####################################
with tf.name_scope('fc8_VividColor') as scope:
    net_weights = net.params['fc8_VividColor'][0].data
    net_weights = np.transpose(net_weights, (1, 0))
    fc8_VividColor_weights = tf.Variable(net_weights, name="weights")
    biases = tf.Variable(net.params['fc8_VividColor'][1].data, name="bias")
    fullWeights = tf.nn.bias_add(tf.matmul(fc7, net_weights), biases)
    fc8_VividColor = tf.nn.relu(fullWeights)

    net_weights = net.params['fc9_VividColor'][0].data
    net_weights = np.transpose(net_weights, (1, 0))
    fc9_VividColor_weights = tf.Variable(net_weights, name="weights")
    biases = tf.Variable(net.params['fc9_VividColor'][1].data, name="bias")
    fullWeights = tf.nn.bias_add(tf.matmul(fc8_VividColor, net_weights), biases)
    fc9_VividColor = tf.nn.relu(fullWeights)

####################### fc10_merge ##################################
with tf.name_scope('fc10_merge') as scope:
    concat_weights = tf.concat(
        (fc8new,
        fc8_BalancingElement,
        fc8_ColorHarmony,
        fc8_Content,
        fc8_DoF,
        fc8_Light,
        fc8_MotionBlur,
        fc8_Object,
        fc8_Repetition,
        fc8_RuleOfThirds,
        fc8_Symmetry,
        fc8_VividColor),axis=1)

    net_weights = net.params['fc10_merge'][0].data
    net_weights = np.transpose(net_weights, (1, 0))
    fc10_weights = tf.Variable(net_weights, name="weights")
    biases = tf.Variable(net.params['fc10_merge'][1].data, name="bias")
    fullWeights = tf.nn.bias_add(tf.matmul(concat_weights,net_weights),biases)
    fc10_merge = tf.nn.relu(fullWeights)
    print_activations(fc10_merge)
################################# fc7#####################################
with tf.name_scope('fc11_score') as scope:
    net_weights = net.params['fc11_score'][0].data
    net_weights = np.transpose(net_weights,(1,0))
    fc11_score_weights = tf.Variable(net_weights,name="weights")
    biases = tf.Variable(net.params['fc11_score'][1].data,name="bias")
    fullWeights = tf.nn.bias_add(tf.matmul(fc10_merge,net_weights),biases)
    fc11_score = tf.nn.relu(fullWeights)
    print_activations(fc11_score)


##### Run Graph #####
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    output = sess.run([fc11_score],
                      feed_dict={x:images})

