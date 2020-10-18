# coding: utf-8

from __future__ import division, print_function

import numpy as np
import tensorflow as tf
slim = tf.contrib.slim

from inject_util import fp16_dtype_getter

import random

def inj_conv2d(inputs, filters, kernel_size, strides=1, fixed_padding=True, inj_type=None, quant_min_max=None, normalizer_fn=slim.batch_norm, biases_initializer=None, activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=0.1), scope=None, inj_layer=None, delta_4d=None, num_layer=None, batch_norm_params=None):
    def _fixed_padding(inputs, kernel_size):
        pad_total = kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg

        padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                        [pad_beg, pad_end], [0, 0]], mode='CONSTANT')
        return padded_inputs
    if strides > 1 and fixed_padding: 
        inputs = _fixed_padding(inputs, kernel_size)

    conv_scope = 'Conv' + ('' if num_layer == 0 else '_' + str(num_layer))

    with tf.variable_scope(conv_scope, fp16_dtype_getter):
    #with tf.variable_scope(conv_scope):
        inputs = tf.cast(inputs, tf.float16)
        if normalizer_fn is not None:
            conv_out = slim.conv2d(inputs, filters, kernel_size, stride=strides, normalizer_fn=None, biases_initializer=None,
                         padding=('SAME' if strides == 1 else 'VALID'), activation_fn=None)

        else:
            conv_out = slim.conv2d(inputs, filters, kernel_size, stride=strides, normalizer_fn=None, biases_initializer=biases_initializer,
                         padding=('SAME' if strides == 1 else 'VALID'), activation_fn=None)
        conv_out = tf.cast(conv_out, tf.float32)

        if normalizer_fn is not None:
            conv_out = normalizer_fn(conv_out, **batch_norm_params)

    # If this is the inject layer
    if inj_layer and len(inj_layer) > 1:
        print('Error: Do not support injecting to more than 1 layer!')
        exit(12)

    if inj_layer:
        target_layer_compare = inj_layer[0]
        top_name_scope = tf.get_default_graph().get_name_scope()

        top_name_scope = top_name_scope + '/Conv' + ('_{}'.format(num_layer) if num_layer != 0 else '') + '/Conv/Conv2D'

        #print('Target layer compare is {}'.format(target_layer_compare))
        #print('Top name scope is {}'.format(top_name_scope))

        if target_layer_compare and top_name_scope == target_layer_compare:
            print('Getting into inject layer')
            if inj_type[:inj_type.find('_')] in ['RD', 'INPUT', 'INPUT_16', 'WEIGHT', 'WEIGHT_16', 'RD_BFLIP', 'RD_PSUM']:
                print('Performing delta addition')
                conv_out = tf.add(conv_out, delta_4d)
            else:
                print('Invalid parameter!')
                exit(12)

    # Then do clipping if needed
    if inj_type is not None:
        # First solve all the NaNs - Set them to 0
        conv_out = tf.where(tf.is_nan(conv_out), tf.ones_like(conv_out) * 0, conv_out)
        if 'F32' in inj_type:
            clipped_conv_out = tf.clip_by_value(conv_out, -3.402823e38, 3.402823e38)
        elif 'F16' in inj_type:
            clipped_conv_out = tf.clip_by_value(conv_out, -65504, 65504)
        elif is_inject and ('I16' in inj_type or 'I8' in inj_type) and ('INPUT' in inj_type or 'WEIGHT' in inj_type or 'PSUM' in inj_type):
            clipped_conv_out = tf.clip_by_value(conv_out, quant_min_max[0], quant_min_max[1])
        else:
            clipped_conv_out = conv_out
    else:
        clipped_conv_out = conv_out

    # Perform batch norm and actiation
    if activation_fn is not None:
        relu_out = activation_fn(clipped_conv_out)
    else:
        relu_out = clipped_conv_out

    return relu_out


def darknet53_body(inputs, inj_type=None, quant_min_max=None, inj_layer=None, delta_4d=None, batch_norm_params=None):
    num_layer = 0
    def res_block(inputs, filters, inj_type=None, quant_min_max=None, inj_layer=None, delta_4d=None, num_layer=num_layer):
        shortcut = inputs
        net = inj_conv2d(inputs, filters * 1, 1, inj_type=inj_type, quant_min_max=quant_min_max, inj_layer=inj_layer, delta_4d=delta_4d, num_layer=num_layer, batch_norm_params=batch_norm_params)

        num_layer += 1
        net = inj_conv2d(net, filters * 2, 3, inj_type=inj_type, quant_min_max=quant_min_max, inj_layer=inj_layer, delta_4d=delta_4d, num_layer=num_layer, batch_norm_params=batch_norm_params)

        net = net + shortcut

        return net
    
    # first two conv2d layers
    net = inj_conv2d(inputs, 32,  3, strides=1, inj_type=inj_type, quant_min_max=quant_min_max, inj_layer=inj_layer, delta_4d=delta_4d, num_layer=num_layer, batch_norm_params=batch_norm_params)
    num_layer += 1
    net = inj_conv2d(net, 64,  3, strides=2, inj_type=inj_type, quant_min_max=quant_min_max, inj_layer=inj_layer, delta_4d=delta_4d, num_layer=num_layer, batch_norm_params=batch_norm_params)
    num_layer += 1

    # res_block * 1
    net = res_block(net, 32, inj_type=inj_type, quant_min_max=quant_min_max, inj_layer=inj_layer, delta_4d=delta_4d, num_layer=num_layer)
    num_layer += 2

    net = inj_conv2d(net, 128, 3, strides=2, inj_type=inj_type, quant_min_max=quant_min_max, inj_layer=inj_layer, delta_4d=delta_4d, num_layer=num_layer, batch_norm_params=batch_norm_params)
    num_layer += 1

    # res_block * 2
    for i in range(2):
        net = res_block(net, 64, inj_type=inj_type, quant_min_max=quant_min_max, inj_layer=inj_layer, delta_4d=delta_4d, num_layer=num_layer)
        num_layer += 2

    net = inj_conv2d(net, 256, 3, strides=2, inj_type=inj_type, quant_min_max=quant_min_max, inj_layer=inj_layer, delta_4d=delta_4d, num_layer=num_layer, batch_norm_params=batch_norm_params)
    num_layer += 1

    # res_block * 8
    for i in range(8):
        net = res_block(net, 128, inj_type=inj_type, quant_min_max=quant_min_max, inj_layer=inj_layer, delta_4d=delta_4d, num_layer=num_layer)
        num_layer += 2

    route_1 = net
    net = inj_conv2d(net, 512, 3, strides=2, inj_type=inj_type, quant_min_max=quant_min_max, inj_layer=inj_layer, delta_4d=delta_4d, num_layer=num_layer, batch_norm_params=batch_norm_params)
    num_layer += 1

    # res_block * 8
    for i in range(8):
        net = res_block(net, 256, inj_type=inj_type, quant_min_max=quant_min_max, inj_layer=inj_layer, delta_4d=delta_4d, num_layer=num_layer)
        num_layer += 2

    route_2 = net
    net = inj_conv2d(net, 1024, 3, strides=2, inj_type=inj_type, quant_min_max=quant_min_max, inj_layer=inj_layer, delta_4d=delta_4d, num_layer=num_layer, batch_norm_params=batch_norm_params)
    num_layer += 1

    # res_block * 4
    for i in range(4):
        net = res_block(net, 512, inj_type=inj_type, quant_min_max=quant_min_max, inj_layer=inj_layer, delta_4d=delta_4d, num_layer=num_layer)
        num_layer += 2
    route_3 = net

    return route_1, route_2, route_3


def yolo_block(inputs, filters, inj_type=None, quant_min_max=None, inj_layer=None, delta_4d=None, num_layer=None, batch_norm_params=None):
    net = inj_conv2d(inputs, filters * 1, 1, inj_type=inj_type, quant_min_max=quant_min_max, inj_layer=inj_layer, delta_4d=delta_4d, num_layer=num_layer, batch_norm_params=batch_norm_params)
    num_layer += 1
    net = inj_conv2d(net, filters * 2, 3, inj_type=inj_type, quant_min_max=quant_min_max, inj_layer=inj_layer, delta_4d=delta_4d, num_layer=num_layer, batch_norm_params=batch_norm_params)
    num_layer += 1
    net = inj_conv2d(net, filters * 1, 1, inj_type=inj_type, quant_min_max=quant_min_max, inj_layer=inj_layer, delta_4d=delta_4d, num_layer=num_layer, batch_norm_params=batch_norm_params)
    num_layer += 1
    net = inj_conv2d(net, filters * 2, 3, inj_type=inj_type, quant_min_max=quant_min_max, inj_layer=inj_layer, delta_4d=delta_4d, num_layer=num_layer, batch_norm_params=batch_norm_params)
    num_layer += 1
    net = inj_conv2d(net, filters * 1, 1, inj_type=inj_type, quant_min_max=quant_min_max, inj_layer=inj_layer, delta_4d=delta_4d, num_layer=num_layer, batch_norm_params=batch_norm_params)
    num_layer += 1
    route = net
    net = inj_conv2d(net, filters * 2, 3, inj_type=inj_type, quant_min_max=quant_min_max, inj_layer=inj_layer, delta_4d=delta_4d, num_layer=num_layer, batch_norm_params=batch_norm_params)
    num_layer += 1
    return route, net


def upsample_layer(inputs, out_shape):
    new_height, new_width = out_shape[1], out_shape[2]
    # NOTE: here height is the first
    # TODO: Do we need to set `align_corners` as True?
    inputs = tf.image.resize_nearest_neighbor(inputs, (new_height, new_width), name='upsampled')
    return inputs

