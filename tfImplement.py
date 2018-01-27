import numpy as np
import  tensorflow as tf


WEIGHT_DECAY_KEY = 'WEIGHT_DECAY'
LogDir = "/home/manish/projects/objectiveTF/logDir"

import torch
import torch.nn as nn
import torchvision.models as models



resnet = models.resnet18(pretrained=True)
model_dict  = resnet.state_dict()


def _relu(x, leakness=0.0, name=None):
    name = 'relu' if name is None else 'lrelu'
    if leakness > 0.0:
        return tf.maximum(x, x*leakness, name=name)
    else:
        return tf.nn.relu(x, name=name)

def _conv(x, kernel, strides, pad='SAME', name='conv'):
    conv = tf.nn.conv2d(x, kernel, [1, strides, strides, 1], pad)
    return conv


def _bn(x,  mean, var, beta, gamma, global_step=None, name='bn'):
    moving_average_decay = 0.9
    with tf.variable_scope(name):
        decay = moving_average_decay

        bn = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-5)

    return bn


def _fc():
    weights = model_dict['fc.weight'].numpy()
    weights = np.transpose(weights)
    wt = tf.Variable(weights,name="weight")
    bias = model_dict['fc.bias'].numpy()
    bs = tf.Variable(bias,name="bias")
    return (wt,bs)

def _bottleNeck(input, wtDict, strides, name="unit"):
        # Shortcut connection
    print('\tBuilding residual unit: %s' % name)
    kernel, gamma, beta, mean, var = wtDict[3]
    residual = _conv(input, kernel, strides,pad='VALID', name='residual')
    residual = _bn(residual, mean, var, beta, gamma, name='bn_res')
    residual = _relu(residual, name='relu_res')
    # Residual
    kernel, gamma, beta, mean, var = wtDict[1]
    input = _conv(input, kernel, 2, name='conv_1')
    input = _bn(input, mean, var, beta, gamma, name='bn_1')
    input = _relu(input, name='relu_1')

    kernel, gamma, beta, mean, var = wtDict[2]
    input = _conv(input, kernel, 1, name='conv_2')
    input = _bn(input, mean, var, beta, gamma, name='bn_2')
    # Merge
    input = input + residual
    input = _relu(input, name='relu_2')
    return input


def _basicBlock(input, wtDict,  name="unit"):

    print('\tBuilding residual unit: %s' % name)
    # Shortcut connection
    residual = input
    # Residual
    kernel, gamma, beta, mean, var = wtDict[1]
    input = _conv(input, kernel, 1,  name='conv_1')
    input = _bn(input, mean, var, beta, gamma, name='bn_1')
    input = _relu(input, name='relu_1')

    kernel, gamma, beta, mean, var = wtDict[2]
    input = _conv(input, kernel, 1, name='conv_2')
    input = _bn(input, mean, var, beta, gamma, name='bn_2')

    input = input + residual
    input = _relu(input, name='relu_2')
    return input


params = {}
conv = "conv{}.weight".format
bn = "bn{}.weight".format
bias ="bn{}.bias".format
mean = "bn{}.running_mean".format
var = "bn{}.running_var".format

Param = [conv, bn, bias, mean, var]


ParamDown = ['weight', 'weight', 'bias', 'running_mean', 'running_var']

def get_tf_variable(layer=None, level=1):
    weights = []
    for i,p in enumerate(Param):
        key = "layer"+layer+'.'+p(level) if layer is not None else p(level)
        wt = model_dict[key].numpy()
        if wt.ndim > 1:
            wt = np.transpose(wt, (2, 3, 1, 0))
        tf_wt = tf.Variable(wt,name=p(level))
        weights.append(tf_wt)
    return weights

def get_downsample_variable(layer):
    weights = []
    layer += '.downsample'
    for i,p in enumerate(ParamDown):
        ind = 1 if i != 0 else 0
        key = "layer"+layer+'.'+str(ind)+'.'+p
        wt = model_dict[key].numpy()
        if wt.ndim > 1:
            wt = np.transpose(wt, (2, 3, 1, 0))
        tf_wt = kernel = tf.Variable(wt,name='downSample'+p)
        weights.append(tf_wt)
    return weights


######################################## Model Definition #############################

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
x = tf.placeholder(tf.float32,(None,IMAGE_WIDTH,IMAGE_HEIGHT,3))

with tf.name_scope('conv1') as scope:
    kernel,gamma,beta,mean,var = get_tf_variable()
    conImg = _conv(x, kernel, 2)
    bnorm = _bn(conImg,mean,var,beta,gamma)
    relu = _relu(bnorm)
    pool1 = tf.nn.max_pool(relu, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME')


with tf.name_scope('layer1') as scope:
    layer = '1.0'
    wtLst = {}
    for lv in range(1,3):
        wtLst[lv] = get_tf_variable(layer, lv)
    layer1_0 = _basicBlock(pool1,wtLst,name=layer)

    layer = '1.1'
    wtLst = {}
    for lv in range(1, 3):
        wtLst[lv] = get_tf_variable(layer, lv)
    layer1_1 = _basicBlock(layer1_0, wtLst, name=layer)

with tf.name_scope('layer2') as scope:
    layer = '2.0'
    wtLst = {}
    for lv in range(1,3):
        wtLst[lv] = get_tf_variable(layer, lv)
    wtLst[len(wtLst)+1] = get_downsample_variable(layer)

    layer2_0 = _bottleNeck(layer1_1,wtLst,2,name=layer)

    layer = '2.1'
    wtLst = {}
    for lv in range(1, 3):
        wtLst[lv] = get_tf_variable(layer, lv)
    layer2_1 = _basicBlock(layer2_0, wtLst, name=layer)

with tf.name_scope('layer3') as scope:
    layer = '3.0'
    wtLst = {}
    for lv in range(1,3):
        wtLst[lv] = get_tf_variable(layer, lv)
    wtLst[len(wtLst)+1] = get_downsample_variable(layer)

    layer3_0 = _bottleNeck(layer2_1,wtLst,2,name=layer)

    layer = '3.1'
    wtLst = {}
    for lv in range(1, 3):
        wtLst[lv] = get_tf_variable(layer, lv)
    layer3_1 = _basicBlock(layer3_0, wtLst, name=layer)

with tf.name_scope('layer4') as scope:
    layer = '4.0'
    wtLst = {}
    for lv in range(1,3):
        wtLst[lv] = get_tf_variable(layer, lv)
    wtLst[len(wtLst)+1] = get_downsample_variable(layer)

    layer4_0 = _bottleNeck(layer3_1,wtLst,2,name=layer)

    layer = '4.1'
    wtLst = {}
    for lv in range(1, 3):
        wtLst[lv] = get_tf_variable(layer, lv)
    layer4_1 = _basicBlock(layer4_0, wtLst, name=layer)

with tf.name_scope('fc') as scope:
    print('\tBuilding unit: %s' % scope)
    fc6 = tf.reduce_mean(layer4_1, [1, 2])
    wt,bias = _fc()
    logits = tf.nn.bias_add(tf.matmul(fc6, wt), bias)

    probs = tf.nn.softmax(logits)
    preds = tf.to_int32(tf.argmax(logits, 1))




#### Image Preparation#########################
import cv2

files = ["/home/manish/projects/objectiveTF/Data/img10.jpeg","/home/manish/projects/objectiveTF/Data/img12.jpeg"]
images = np.zeros((len(files),IMAGE_HEIGHT,IMAGE_WIDTH,3))
for i,f in enumerate(files):
    img = cv2.imread(f)
    img = cv2.resize(img,(224,224),interpolation=cv2.INTER_LINEAR)
    images[i, :, :, :] = img



init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    summary_writer = tf.summary.FileWriter(LogDir, sess.graph)
    output = sess.run([preds],
                      feed_dict={x:images})

    print (output)