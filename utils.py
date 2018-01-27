CAFFE_ROOT = '/home/manish/caffe-rc5/'
import sys
sys.path.insert(0, CAFFE_ROOT + 'python')

import numpy as np
import skimage.io
from scipy.ndimage import zoom
from skimage.transform import resize

try:
    # Python3 will most likely not be able to load protobuf
    from caffe.proto import caffe_pb2
except:
    import sys
    if sys.version_info >= (3, 0):
        print("Failed to include caffe_pb2, things might go wrong!")
    else:
        raise


## proto / datum / ndarray conversion
def blobproto_to_array(blob, return_diff=False):
    """
    Convert a blob proto to an array. In default, we will just return the data,
    unless return_diff is True, in which case we will return the diff.
    """
    # Read the data into an array
    if return_diff:
        data = np.array(blob.diff)
    else:
        data = np.array(blob.data)

    # Reshape the array
    if blob.HasField('num') or blob.HasField('channels') or blob.HasField('height') or blob.HasField('width'):
        # Use legacy 4D shape
        return data.reshape(blob.num, blob.channels, blob.height, blob.width)
    else:
        return data.reshape(blob.shape.dim)

def array_to_blobproto(arr, diff=None):
    """Converts a N-dimensional array to blob proto. If diff is given, also
    convert the diff. You need to make sure that arr and diff have the same
    shape, and this function does not do sanity check.
    """
    blob = caffe_pb2.BlobProto()
    blob.shape.dim.extend(arr.shape)
    blob.data.extend(arr.astype(float).flat)
    if diff is not None:
        blob.diff.extend(diff.astype(float).flat)
    return blob


def arraylist_to_blobprotovector_str(arraylist):
    """Converts a list of arrays to a serialized blobprotovec, which could be
    then passed to a network for processing.
    """
    vec = caffe_pb2.BlobProtoVector()
    vec.blobs.extend([array_to_blobproto(arr) for arr in arraylist])
    return vec.SerializeToString()


def blobprotovector_str_to_arraylist(str):
    """Converts a serialized blobprotovec to a list of arrays.
    """
    vec = caffe_pb2.BlobProtoVector()
    vec.ParseFromString(str)
    return [blobproto_to_array(blob) for blob in vec.blobs]


def array_to_datum(arr, label=None):
    """Converts a 3-dimensional array to datum. If the array has dtype uint8,
    the output data will be encoded as a string. Otherwise, the output data
    will be stored in float format.
    """
    if arr.ndim != 3:
        raise ValueError('Incorrect array shape.')
    datum = caffe_pb2.Datum()
    datum.channels, datum.height, datum.width = arr.shape
    if arr.dtype == np.uint8:
        datum.data = arr.tostring()
    else:
        datum.float_data.extend(arr.flat)
    if label is not None:
        datum.label = label
    return datum


def datum_to_array(datum):
    """Converts a datum to an array. Note that the label is not returned,
    as one can easily get it by calling datum.label.
    """
    if len(datum.data):
        return np.fromstring(datum.data, dtype=np.uint8).reshape(
            datum.channels, datum.height, datum.width)
    else:
        return np.array(datum.float_data).astype(float).reshape(
            datum.channels, datum.height, datum.width)


## Pre-processing

class Transformer:
    """
    Transform input for feeding into a Net.

    Note: this is mostly for illustrative purposes and it is likely better
    to define your own input preprocessing routine for your needs.

    Parameters
    ----------
    net : a Net for which the input should be prepared
    """
    def __init__(self, inputs):
        self.inputs = inputs
        self.transpose = {}
        self.channel_swap = {}
        self.raw_scale = {}
        self.mean = {}
        self.input_scale = {}

    def __check_input(self, in_):
        if in_ not in self.inputs:
            raise Exception('{} is not one of the net inputs: {}'.format(
                in_, self.inputs))

    def preprocess(self, in_, data):
        """
        Format input for Caffe:
        - convert to single
        - resize to input dimensions (preserving number of channels)
        - transpose dimensions to K x H x W
        - reorder channels (for instance color to BGR)
        - scale raw input (e.g. from [0, 1] to [0, 255] for ImageNet models)
        - subtract mean
        - scale feature

        Parameters
        ----------
        in_ : name of input blob to preprocess for
        data : (H' x W' x K) ndarray

        Returns
        -------
        caffe_in : (K x H x W) ndarray for input to a Net
        """
        self.__check_input(in_)
        caffe_in = data.astype(np.float32, copy=False)
        transpose = self.transpose.get(in_)
        channel_swap = self.channel_swap.get(in_)
        raw_scale = self.raw_scale.get(in_)
        mean = self.mean.get(in_)
        input_scale = self.input_scale.get(in_)
        in_dims = self.inputs[in_][2:]
        if caffe_in.shape[:2] != in_dims:
            caffe_in = resize_image(caffe_in, in_dims)
        if transpose is not None:
            caffe_in = caffe_in.transpose(transpose)
        if channel_swap is not None:
            caffe_in = caffe_in[channel_swap, :, :]
        if raw_scale is not None:
            caffe_in *= raw_scale
        if mean is not None:
            caffe_in -= mean
        if input_scale is not None:
            caffe_in *= input_scale
        return caffe_in

    def deprocess(self, in_, data):
        """
        Invert Caffe formatting; see preprocess().
        """
        self.__check_input(in_)
        decaf_in = data.copy().squeeze()
        transpose = self.transpose.get(in_)
        channel_swap = self.channel_swap.get(in_)
        raw_scale = self.raw_scale.get(in_)
        mean = self.mean.get(in_)
        input_scale = self.input_scale.get(in_)
        if input_scale is not None:
            decaf_in /= input_scale
        if mean is not None:
            decaf_in += mean
        if raw_scale is not None:
            decaf_in /= raw_scale
        if channel_swap is not None:
            decaf_in = decaf_in[np.argsort(channel_swap), :, :]
        if transpose is not None:
            decaf_in = decaf_in.transpose(np.argsort(transpose))
        return decaf_in

    def set_transpose(self, in_, order):
        """
        Set the input channel order for e.g. RGB to BGR conversion
        as needed for the reference ImageNet model.

        Parameters
        ----------
        in_ : which input to assign this channel order
        order : the order to transpose the dimensions
        """
        self.__check_input(in_)
        if len(order) != len(self.inputs[in_]) - 1:
            raise Exception('Transpose order needs to have the same number of '
                            'dimensions as the input.')
        self.transpose[in_] = order

    def set_channel_swap(self, in_, order):
        """
        Set the input channel order for e.g. RGB to BGR conversion
        as needed for the reference ImageNet model.
        N.B. this assumes the channels are the first dimension AFTER transpose.

        Parameters
        ----------
        in_ : which input to assign this channel order
        order : the order to take the channels.
            (2,1,0) maps RGB to BGR for example.
        """
        self.__check_input(in_)
        if len(order) != self.inputs[in_][1]:
            raise Exception('Channel swap needs to have the same number of '
                            'dimensions as the input channels.')
        self.channel_swap[in_] = order

    def set_raw_scale(self, in_, scale):
        """
        Set the scale of raw features s.t. the input blob = input * scale.
        While Python represents images in [0, 1], certain Caffe models
        like CaffeNet and AlexNet represent images in [0, 255] so the raw_scale
        of these models must be 255.

        Parameters
        ----------
        in_ : which input to assign this scale factor
        scale : scale coefficient
        """
        self.__check_input(in_)
        self.raw_scale[in_] = scale

    def set_mean(self, in_, mean):
        """
        Set the mean to subtract for centering the data.

        Parameters
        ----------
        in_ : which input to assign this mean.
        mean : mean ndarray (input dimensional or broadcastable)
        """
        self.__check_input(in_)
        ms = mean.shape
        if mean.ndim == 1:
            # broadcast channels
            if ms[0] != self.inputs[in_][1]:
                raise ValueError('Mean channels incompatible with input.')
            mean = mean[:, np.newaxis, np.newaxis]
        else:
            # elementwise mean
            if len(ms) == 2:
                ms = (1,) + ms
            if len(ms) != 3:
                raise ValueError('Mean shape invalid')
            if ms != self.inputs[in_][1:]:
                raise ValueError('Mean shape incompatible with input shape.')
        self.mean[in_] = mean

    def set_input_scale(self, in_, scale):
        """
        Set the scale of preprocessed inputs s.t. the blob = blob * scale.
        N.B. input_scale is done AFTER mean subtraction and other preprocessing
        while raw_scale is done BEFORE.

        Parameters
        ----------
        in_ : which input to assign this scale factor
        scale : scale coefficient
        """
        self.__check_input(in_)
        self.input_scale[in_] = scale


## Image IO

def load_image(filename, color=True):
    """
    Load an image converting from grayscale or alpha as needed.

    Parameters
    ----------
    filename : string
    color : boolean
        flag for color format. True (default) loads as RGB while False
        loads as intensity (if image is already grayscale).

    Returns
    -------
    image : an image with type np.float32 in range [0, 1]
        of size (H x W x 3) in RGB or
        of size (H x W x 1) in grayscale.
    """
    img = skimage.img_as_float(skimage.io.imread(filename, as_grey=not color)).astype(np.float32)
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
        if color:
            img = np.tile(img, (1, 1, 3))
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    return img


def resize_image(im, new_dims, interp_order=1):
    """
    Resize an image array with interpolation.

    Parameters
    ----------
    im : (H x W x K) ndarray
    new_dims : (height, width) tuple of new dimensions.
    interp_order : interpolation order, default is linear.

    Returns
    -------
    im : resized ndarray with shape (new_dims[0], new_dims[1], K)
    """
    if im.shape[-1] == 1 or im.shape[-1] == 3:
        im_min, im_max = im.min(), im.max()
        if im_max > im_min:
            # skimage is fast but only understands {1,3} channel images
            # in [0, 1].
            im_std = (im - im_min) / (im_max - im_min)
            resized_std = resize(im_std, new_dims, order=interp_order)
            resized_im = resized_std * (im_max - im_min) + im_min
        else:
            # the image is a constant -- avoid divide by 0
            ret = np.empty((new_dims[0], new_dims[1], im.shape[-1]),
                           dtype=np.float32)
            ret.fill(im_min)
            return ret
    else:
        # ndimage interpolates anything but more slowly.
        scale = tuple(np.array(new_dims, dtype=float) / np.array(im.shape[:2]))
        resized_im = zoom(im, scale + (1,), order=interp_order)
    return resized_im.astype(np.float32)


def oversample(images, crop_dims):
    """
    Crop images into the four corners, center, and their mirrored versions.

    Parameters
    ----------
    image : iterable of (H x W x K) ndarrays
    crop_dims : (height, width) tuple for the crops.

    Returns
    -------
    crops : (10*N x H x W x K) ndarray of crops for number of inputs N.
    """
    # Dimensions and center.
    im_shape = np.array(images[0].shape)
    crop_dims = np.array(crop_dims)
    im_center = im_shape[:2] / 2.0

    # Make crop coordinates
    h_indices = (0, im_shape[0] - crop_dims[0])
    w_indices = (0, im_shape[1] - crop_dims[1])
    crops_ix = np.empty((5, 4), dtype=int)
    curr = 0
    for i in h_indices:
        for j in w_indices:
            crops_ix[curr] = (i, j, i + crop_dims[0], j + crop_dims[1])
            curr += 1
    crops_ix[4] = np.tile(im_center, (1, 2)) + np.concatenate([
        -crop_dims / 2.0,
         crop_dims / 2.0
    ])
    crops_ix = np.tile(crops_ix, (2, 1))

    # Extract crops
    crops = np.empty((10 * len(images), crop_dims[0], crop_dims[1],
                      im_shape[-1]), dtype=np.float32)
    ix = 0
    for im in images:
        for crop in crops_ix:
            crops[ix] = im[crop[0]:crop[2], crop[1]:crop[3], :]
            ix += 1
        crops[ix-5:ix] = crops[ix-5:ix, :, ::-1, :]  # flip for mirrors
    return crops


import tensorflow as tf

## TensorFlow helper functions

WEIGHT_DECAY_KEY = 'WEIGHT_DECAY'

def print_activations(t):
  print(t.op.name, ' ', t.get_shape().as_list())

def _relu(x, leakness=0.0, name=None):
    if leakness > 0.0:
        name = 'lrelu' if name is None else name
        return tf.maximum(x, x*leakness, name='lrelu')
    else:
        name = 'relu' if name is None else name
        return tf.nn.relu(x, name='relu')


def _conv(x, filter_size, out_channel, strides, pad='SAME', input_q=None, output_q=None, name='conv'):
    if (input_q == None)^(output_q == None):
        raise ValueError('Input/Output splits are not correctly given.')

    in_shape = x.get_shape()
    with tf.variable_scope(name):
        # Main operation: conv2d
        with tf.device('/CPU:0'):
            kernel = tf.get_variable('kernel', [filter_size, filter_size, in_shape[3], out_channel],
                            tf.float32, initializer=tf.random_normal_initializer(
                                stddev=np.sqrt(2.0/filter_size/filter_size/out_channel)))
        if kernel not in tf.get_collection(WEIGHT_DECAY_KEY):
            tf.add_to_collection(WEIGHT_DECAY_KEY, kernel)
            # print('\tadded to WEIGHT_DECAY_KEY: %s(%s)' % (kernel.name, str(kernel.get_shape().as_list())))
        conv = tf.nn.conv2d(x, kernel, [1, strides, strides, 1], pad)

        # Split and split loss
        if (input_q is not None) and (output_q is not None):
            # w = tf.reduce_mean(kernel, axis=[0, 1])
            # w = tf.sqrt(tf.reduce_mean(tf.square(kernel), [0, 1]))
            _add_split_loss(kernel, input_q, output_q)

    return conv


def _fc(x, out_dim, input_q=None, output_q=None, name='fc'):
    if (input_q == None)^(output_q == None):
        raise ValueError('Input/Output splits are not correctly given.')

    with tf.variable_scope(name):
        # Main operation: fc
        with tf.device('/CPU:0'):
            w = tf.get_variable('weights', [x.get_shape()[1], out_dim],
                            tf.float32, initializer=tf.random_normal_initializer(
                                stddev=np.sqrt(1.0/out_dim)))
            b = tf.get_variable('biases', [out_dim], tf.float32,
                                initializer=tf.constant_initializer(0.0))
        if w not in tf.get_collection(WEIGHT_DECAY_KEY):
            tf.add_to_collection(WEIGHT_DECAY_KEY, w)
            # print('\tadded to WEIGHT_DECAY_KEY: %s(%s)' % (w.name, str(w.get_shape().as_list())))
        fc = tf.nn.bias_add(tf.matmul(x, w), b)

        # Split loss
        if (input_q is not None) and (output_q is not None):
            _add_split_loss(w, input_q, output_q)

    return fc


def _get_split_q(ngroups, dim, name='split', l2_loss=False):
    with tf.variable_scope(name):
        # alpha = tf.get_variable('alpha', shape=[ngroups, dim], dtype=tf.float32,
                              # initializer=tf.random_normal_initializer(stddev=0.1))
        # q = tf.nn.softmax(alpha, dim=0, name='q')
        std_dev = 0.01
        init_val = np.random.normal(0, std_dev, (ngroups, dim))
        init_val = init_val - np.average(init_val, axis=0) + 1.0/ngroups
        with tf.device('/CPU:0'):
            q = tf.get_variable('q', shape=[ngroups, dim], dtype=tf.float32,
                                # initializer=tf.constant_initializer(1.0/ngroups))
                                initializer=tf.constant_initializer(init_val))
        if l2_loss:
            if q not in tf.get_collection(WEIGHT_DECAY_KEY):
                tf.add_to_collection(WEIGHT_DECAY_KEY, q*2.236)

    return q

def _merge_split_q(q, merge_idxs, name='merge'):
    assert len(q.get_shape()) == 2
    ngroups, dim = q.get_shape().as_list()
    assert ngroups == len(merge_idxs)

    with tf.variable_scope(name):
        max_idx = np.max(merge_idxs)
        temp_list = []
        for i in range(max_idx + 1):
            temp = []
            for j in range(ngroups):
                if merge_idxs[j] == i:
                    temp.append(tf.slice(q, [j, 0], [1, dim]))
            temp_list.append(tf.add_n(temp))
        ret = tf.concat(0, temp_list)

    return ret


def _get_even_merge_idxs(N, split):
    assert N >= split
    num_elems = [(N + split - i - 1)/split for i in range(split)]
    expand_split = [[i] * n for i, n in enumerate(num_elems)]
    return [t for l in expand_split for t in l]


def _add_split_loss(w, input_q, output_q):
    # Check input tensors' measurements
    assert len(w.get_shape()) == 2 or len(w.get_shape()) == 4
    in_dim, out_dim = w.get_shape().as_list()[-2:]
    assert len(input_q.get_shape()) == 2
    assert len(output_q.get_shape()) == 2
    assert in_dim == input_q.get_shape().as_list()[1]
    assert out_dim == output_q.get_shape().as_list()[1]
    assert input_q.get_shape().as_list()[0] == output_q.get_shape().as_list()[0]  # ngroups
    ngroups = input_q.get_shape().as_list()[0]
    assert ngroups > 1

    # Add split losses to collections
    T_list = []
    U_list = []
    if input_q not in tf.get_collection('OVERLAP_LOSS_WEIGHTS'):
        tf.add_to_collection('OVERLAP_LOSS_WEIGHTS', input_q)
        print('\t\tAdd overlap & split loss for %s' % input_q.name)
        for i in range(ngroups):
            for j in range(ngroups):
                if i == j:
                    continue
                T_list.append(tf.reduce_sum(input_q[i,:] * input_q[j,:]))
            U_list.append(tf.square(tf.reduce_sum(input_q[i,:])))
    if output_q not in tf.get_collection('OVERLAP_LOSS_WEIGHTS'):
        print('\t\tAdd overlap & split loss for %s' % output_q.name)
        tf.add_to_collection('OVERLAP_LOSS_WEIGHTS', output_q)
        for i in range(ngroups):
            for j in range(ngroups):
                if i == j:
                    continue
                T_list.append(tf.reduce_sum(output_q[i,:] * output_q[j,:]))
            U_list.append(tf.square(tf.reduce_sum(output_q[i,:])))
    if T_list:
        tf.add_to_collection('OVERLAP_LOSS', tf.add_n(T_list))
    if U_list:
        tf.add_to_collection('UNIFORM_LOSS', tf.add_n(U_list))

    S_list = []
    for i in range(ngroups):
        if len(w.get_shape()) == 4:
            w_reduce = tf.reduce_mean(tf.square(w), [0, 1])
            wg_row = tf.matmul(tf.matmul(tf.diag(tf.square(1 - input_q[i,:])), w_reduce), tf.diag(tf.square(output_q[i,:])))
            wg_row_l2 = tf.reduce_sum(tf.sqrt(tf.reduce_sum(wg_row, 1)))
            wg_col = tf.matmul(tf.matmul(tf.diag(tf.square(input_q[i,:])), w_reduce), tf.diag(tf.square(1 - output_q[i,:])))
            wg_col_l2 = tf.reduce_sum(tf.sqrt(tf.reduce_sum(wg_col, 0)))
        else:  # len(w.get_shape()) == 2
            wg_row = tf.matmul(tf.matmul(tf.diag(1 - input_q[i,:]), w), tf.diag(output_q[i,:]))
            wg_row_l2 = tf.reduce_sum(tf.sqrt(tf.reduce_sum(wg_row * wg_row, 1)))
            wg_col = tf.matmul(tf.matmul(tf.diag(input_q[i,:]), w), tf.diag(1 - output_q[i,:]))
            wg_col_l2 = tf.reduce_sum(tf.sqrt(tf.reduce_sum(wg_col * wg_col, 0)))
        S_list.append(wg_row_l2 + wg_col_l2)
    S = tf.add_n(S_list)
    tf.add_to_collection('WEIGHT_SPLIT', S)

    # Add histogram for w if split losses are added
    scope_name = tf.get_variable_scope().name
    tf.histogram_summary("%s/weights" % scope_name, w)
    print('\t\tAdd split loss for %s(%dx%d, %d groups)' \
          % (tf.get_variable_scope().name, in_dim, out_dim, ngroups))

    return


def _bn(x, is_train, global_step=None, name='bn'):
    moving_average_decay = 0.9
    # moving_average_decay = 0.99
    # moving_average_decay_init = 0.99
    with tf.variable_scope(name):
        decay = moving_average_decay
        # if global_step is None:
            # decay = moving_average_decay
        # else:
            # decay = tf.cond(tf.greater(global_step, 100)
                            # , lambda: tf.constant(moving_average_decay, tf.float32)
                            # , lambda: tf.constant(moving_average_decay_init, tf.float32))
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2])
        with tf.device('/CPU:0'):
            mu = tf.get_variable('mu', batch_mean.get_shape(), tf.float32,
                            initializer=tf.zeros_initializer(), trainable=False)
            sigma = tf.get_variable('sigma', batch_var.get_shape(), tf.float32,
                            initializer=tf.ones_initializer(), trainable=False)
            beta = tf.get_variable('beta', batch_mean.get_shape(), tf.float32,
                            initializer=tf.zeros_initializer())
            gamma = tf.get_variable('gamma', batch_var.get_shape(), tf.float32,
                            initializer=tf.ones_initializer())
        # BN when training
        update = 1.0 - decay
        # with tf.control_dependencies([tf.Print(decay, [decay])]):
            # update_mu = mu.assign_sub(update*(mu - batch_mean))
        update_mu = mu.assign_sub(update*(mu - batch_mean))
        update_sigma = sigma.assign_sub(update*(sigma - batch_var))
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_mu)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_sigma)

        mean, var = tf.cond(is_train, lambda: (batch_mean, batch_var),
                            lambda: (mu, sigma))
        bn = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-5)

        # bn = tf.nn.batch_normalization(x, batch_mean, batch_var, beta, gamma, 1e-5)

        # bn = tf.contrib.layers.batch_norm(inputs=x, decay=decay,
                                          # updates_collections=[tf.GraphKeys.UPDATE_OPS], center=True,
                                          # scale=True, epsilon=1e-5, is_training=is_train,
                                          # trainable=True)
    return bn


## Other helper functions


