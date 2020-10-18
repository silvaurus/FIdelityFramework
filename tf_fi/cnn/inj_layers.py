import tensorflow as tf
import numpy as np

slim = tf.contrib.slim

# Creates a fp16 variable scope, such that each variable designed under this scope will be casted to fp16
def custom_dtype_getter(getter, name, shape=None, dtype=tf.float16,
                           *args, **kwargs):
  var = getter(name, shape, tf.float32, *args, **kwargs)
  if 'BatchNorm' in var.name:
    return var
  else:
    return tf.cast(var, dtype=tf.float16, name=name + '_cast')

# Bounding and delta injection for slim.conv2d function
def inj_conv2d(inputs,
                num_outputs,
                kernel_size,
                stride=1,
                padding='SAME',
                rate=1,
                activation_fn=tf.nn.relu,
                normalizer_fn=slim.batch_norm,
                scope=None,
                inj_type=None,
                quant_min_max=None,
                inj_layer=None,
                inj_pos=None):
    top_name_scope = tf.get_default_graph().get_name_scope()
    is_inject = False
    if inj_layer is not None and top_name_scope + '/' + scope in inj_layer:
        is_inject = True
        conv_out = slim.conv2d(inputs, num_outputs, kernel_size, stride=stride, padding=padding, rate=rate, activation_fn=None, normalizer_fn=normalizer_fn, scope=scope)
        
        with tf.name_scope(scope + '_misc'):
            delta_4d = tf.get_variable("delta_{}".format(scope), dtype=tf.float32, shape = conv_out.get_shape().as_list())
            if 'INPUT' in inj_type or 'WEIGHT' in inj_type:
                conv_out = tf.add(conv_out, delta_4d)
            else:
                layer_pos = inj_pos[top_name_scope + '/' + scope]
                num_inj = len(layer_pos)
                mask_conv = np.ones(shape = conv_out.get_shape().as_list(), dtype=np.float32)
                for n_inj in range(num_inj):
                    mask_conv[0][layer_pos[n_inj][0]][layer_pos[n_inj][1]][layer_pos[n_inj][2]] = 0.0
                conv_out = tf.add(tf.multiply(conv_out, mask_conv), delta_4d)
    else:
        conv_out = slim.conv2d(inputs, num_outputs, kernel_size, stride=stride, padding=padding, rate=rate, activation_fn=None, normalizer_fn=normalizer_fn, scope=scope)
         
    if inj_type is not None:
        conv_out = tf.where(tf.is_nan(conv_out), tf.ones_like(conv_out) * 0, conv_out)
        if 'F32' in inj_type:
            clipped_conv_out = tf.clip_by_value(conv_out, -3.402823e38, 3.402823e38)
        elif 'F16' in inj_type:
            clipped_conv_out = tf.clip_by_value(conv_out, -65504, 65504)
        elif is_inject and ('I16' in inj_type or 'I8' in inj_type) and ('INPUT' in inj_type or 'WEIGHT' in inj_type or 'RD_PSUM' in inj_type):
            clipped_conv_out = tf.clip_by_value(conv_out, quant_min_max[0], quant_min_max[1])
        else:
            clipped_conv_out = conv_out
    else:
        clipped_conv_out = conv_out

    if activation_fn is not None:
        relu_out = activation_fn(clipped_conv_out)
    else:
        relu_out = clipped_conv_out
    return relu_out

# inject to slim.separable_conv2d function
def inj_separable_conv2d(
                inputs,
                num_outputs,
                kernel_size,
                depth_multiplier=1,
                stride=1,
                rate=1,
                activation_fn=tf.nn.relu,
                normalizer_fn=slim.batch_norm,
                scope=None,
                inj_type=None,
                quant_min_max=None,
                inj_layer=None,
                inj_pos=None):

    top_name_scope = tf.get_default_graph().get_name_scope()
    is_inject = False

    if inj_layer != None and top_name_scope + '/' + scope in inj_layer:
        is_inject = True
        conv_out = slim.separable_conv2d(inputs, num_outputs, kernel_size, depth_multiplier=depth_multiplier, stride=stride, rate=rate, activation_fn=None, normalizer_fn=normalizer_fn, scope=scope)

        with tf.name_scope(scope + '_misc'):
            delta_4d = tf.get_variable("delta_{}".format(scope), dtype=tf.float32, shape = conv_out.get_shape().as_list())
            if 'INPUT' in inj_type or 'WEIGHT' in inj_type:
                conv_out = tf.add(conv_out, delta_4d)
            else:
                layer_pos = inj_pos[top_name_scope + '/' + scope]
                num_inj = len(layer_pos)
                mask_conv = np.ones(shape = conv_out.get_shape().as_list(), dtype=np.float32)
                for n_inj in range(num_inj):
                    mask_conv[0][layer_pos[n_inj][0]][layer_pos[n_inj][1]][layer_pos[n_inj][2]] = 0.0
                conv_out = tf.add(tf.multiply(conv_out, mask_conv), delta_4d)

    else:
        conv_out = slim.separable_conv2d(inputs, num_outputs, kernel_size, depth_multiplier=depth_multiplier, stride=stride, rate=rate, activation_fn=None, normalizer_fn=normalizer_fn, scope=scope)

    if inj_type is not None:
        conv_out = tf.where(tf.is_nan(conv_out), tf.ones_like(conv_out) * 0, conv_out)
        if 'F32' in inj_type:
            clipped_conv_out = tf.clip_by_value(conv_out, -3.402823e38, 3.402823e38)
        elif 'F16' in inj_type:
            clipped_conv_out = tf.clip_by_value(conv_out, -65504, 65504)
        elif is_inject and ('I16' in inj_type or 'I8' in inj_type) and ('INPUT' in inj_type or 'WEIGHT' in inj_type):
            clipped_conv_out = tf.clip_by_value(conv_out, quant_min_max[0], quant_min_max[1])
        else:
            clipped_conv_out = conv_out
    else:
        clipped_conv_out = conv_out

    if activation_fn is not None:
        relu_out = activation_fn(clipped_conv_out) 
    else:
        relu_out = clipped_conv_out
    return relu_out


# inject to resnet_util.conv2d_same function
def inj_conv2d_same(inputs, num_outputs, kernel_size, stride, rate=1, scope=None, inj_type=None,
                quant_min_max=None,
                inj_layer=None,
                inj_pos=None,
                activation_fn=tf.nn.relu,
                normalizer_fn=slim.batch_norm):
    top_name_scope = tf.get_default_graph().get_name_scope()
    is_inject = False
    if inj_layer is not None and top_name_scope + '/' + scope in inj_layer:
        is_inject = True
    if stride == 1:
        conv_out = slim.conv2d(inputs, num_outputs, kernel_size, stride=1, rate=rate,
                       padding='SAME', scope=scope, activation_fn=None, normalizer_fn=normalizer_fn)
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        inputs = tf.pad(inputs,
                        [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
        
        conv_out = slim.conv2d(inputs, num_outputs, kernel_size, stride=stride,
                       rate=rate, padding='VALID', scope=scope, activation_fn=None, normalizer_fn=normalizer_fn)

    # Then add delta if needed
    if is_inject:
        with tf.name_scope(scope + '_misc'):
            delta_4d = tf.get_variable("delta_{}".format(scope), dtype=tf.float32, shape = conv_out.get_shape().as_list())
            if 'INPUT' in inj_type or 'WEIGHT' in inj_type:
                conv_out = tf.add(conv_out, delta_4d)
            else:
                layer_pos = inj_pos[top_name_scope + '/' + scope]
                num_inj = len(layer_pos)
                mask_conv = np.ones(shape = conv_out.get_shape().as_list(), dtype=np.float32)
                for n_inj in range(num_inj):
                    mask_conv[0][layer_pos[n_inj][0]][layer_pos[n_inj][1]][layer_pos[n_inj][2]] = 0.0
                conv_out = tf.add(tf.multiply(conv_out, mask_conv), delta_4d)
    if inj_type is not None:
        conv_out = tf.where(tf.is_nan(conv_out), tf.ones_like(conv_out) * 0, conv_out)
        if 'F32' in inj_type:
            clipped_conv_out = tf.clip_by_value(conv_out, -3.402823e38, 3.402823e38)
        elif 'F16' in inj_type:
            clipped_conv_out = tf.clip_by_value(conv_out, -65504, 65504)
        elif is_inject and ('I16' in inj_type or 'I8' in inj_type) and ('INPUT' in inj_type or 'WEIGHT' in inj_type):
            clipped_conv_out = tf.clip_by_value(conv_out, quant_min_max[0], quant_min_max[1])
        else:
            clipped_conv_out = conv_out
    else:
        clipped_conv_out = conv_out

    if activation_fn is not None:
        relu_out = activation_fn(clipped_conv_out)
    else:
        relu_out = clipped_conv_out
    return relu_out

# A bounded dense function
def bound_dense(inputs, units, inj_type, kernel_initializer=tf.initializers.truncated_normal, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=4e-05)):
    logits = tf.layers.dense(inputs=inputs, units=units, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)
    if inj_type is not None:
        # First solve all the NaNs - Set them to 0
        logits = tf.where(tf.is_nan(logits), tf.ones_like(logits) * 0, logits)
        if 'F32' in inj_type:
            logits = tf.clip_by_value(logits, -3.402823e38, 3.402823e38)
        elif 'F16' in inj_type:
            logits = tf.clip_by_value(logits, -65504, 65504)
    return logits 

# A bounded squeeze function
def bound_squeeze(inp, axis=None, name=None, inj_type=None):
    sq_result = tf.squeeze(inp, axis=axis, name=name)
    if inj_type is not None:
        # First solve all the NaNs - Set them to 0
        sq_result = tf.where(tf.is_nan(sq_result), tf.ones_like(sq_result) * 0, sq_result)
        if 'F32' in inj_type:
            sq_result = tf.clip_by_value(sq_result, -3.402823e38, 3.402823e38)
        elif 'F16' in inj_type:
            sq_result = tf.clip_by_value(sq_result, -65504, 65504)
    return sq_result


