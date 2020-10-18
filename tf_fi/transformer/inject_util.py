#!/bin/python3
# Written by Yi He
# University of Chicago
# Utilities to support network neuron injections

import tensorflow as tf
import numpy as np
import math
import re

# Converts a fp16 value to binary
def fp162bin(fp):
    sign = math.copysign(1,fp)
    abs_fp = abs(fp)
    # Handling subnormal numbers
    if abs_fp < pow(2,-14):
        target_fp = abs_fp * pow(2,14)
        exponent_bin = '00000'
        frac_bin = ''
        frac_mid = target_fp
        for i in range(25):
            frac_mid *= 2
            if frac_mid >= 1.0:
                frac_bin += '1'
                frac_mid -= 1.0
            else:
                frac_bin += '0'
        mantissa_bin = frac_bin
    # Handling normal numbers
    else:
        int_part = int(np.fix(abs_fp))
        frac_part = abs_fp - int_part
        int_bin = bin(int_part)[2:]
        frac_bin = ''
        frac_mid = frac_part
        for i in range(25):
            frac_mid *= 2
            if frac_mid >= 1.0:
                frac_bin += '1'
                frac_mid -= 1.0
            else:
                frac_bin += '0'
        int_frac_bin = int_bin + frac_bin
        # Decimal point is at the back of variable decimal_point
        decimal_point = len(int_bin)-1
        # Looking for the first 1
        first_one = int_frac_bin.find('1')
        # Special case: 0
        if first_one < 0:
            return ('0x00', '0x00')
        exponent_val = decimal_point - first_one + 15
        assert exponent_val <= 31
        assert exponent_val >= 0
        exponent_bin = bin(exponent_val)[2:].zfill(5)
        mantissa_bin = int_frac_bin[first_one+1:]
        if len(mantissa_bin) < 10:
            mantissa_bin = mantissa_bin.zfill(10)
    if sign == 1.0:
        sign_bin = '0'
    else:
        sign_bin = '1'
    total_bin = (sign_bin + exponent_bin + mantissa_bin)[:16]
    return total_bin


# Converts a binary string to FP16 values
def bin2fp16(bin_str):
    assert len(bin_str) == 16
    sign_bin = bin_str[0]
    if sign_bin == '0':
        sign_val = 1.0
    else:
        sign_val = -1.0
    exponent_bin = bin_str[1:6]
    mantissa_bin = bin_str[6:]
    assert len(mantissa_bin) == 10
    exponent_val = int(exponent_bin,2)
    mantissa_val = 0.0
    for i in range(10):
        if mantissa_bin[i] == '1':
            mantissa_val += pow(2,-i-1)
    # Handling subnormal numbers
    if exponent_val == 0:
        return sign_val * pow(2,-14) * mantissa_val
    # Handling normal numbers
    else:
        value = sign_val * pow(2,exponent_val-15) * (1 + mantissa_val)
        # Handling NaNs and INFs
        if value == 65536:
            return 65535
        elif value == -65536:
            return -65535
        elif value > 65536 or value < -65536:
            return 0
        else:
            return value



# Creates a fp16 variable scope, such that each variable designed under this scope will be casted to fp16
def custom_dtype_getter(getter, name, shape=None, dtype=tf.float16,
                           *args, **kwargs):
    var = getter(name, shape, tf.float32, *args, **kwargs)
    return tf.cast(var, dtype=tf.float16, name=name + '_cast')


def inj_matmul(a, b, inj_type=None, inj_layer=None, scope=None, delta_4d=None, quant_min_max=None):
    with tf.name_scope(scope):
        top_name_scope = tf.get_default_graph().get_name_scope()
        is_inject = False
        matmul_out = tf.matmul(a, b)

        if inj_layer and len(inj_layer) > 1:
            print('Error: Do not support injecting to more than 1 layer!')
            exit(12)

        if inj_layer:
            target_layer_compare = inj_layer[0]
            if target_layer_compare and top_name_scope + '/MatMul' == target_layer_compare:
                is_inject = True
                print('Getting into inject layer')
                # Add delta
                if inj_type[:inj_type.find('_')] in ['RD', 'INPUT', 'INPUT_16', 'WEIGHT', 'WEIGHT_16', 'RD_BFLIP', 'RD_PSUM']:
                    print('Performing delta addition')
                    if delta_4d is not None:
                        matmul_out = tf.add(matmul_out, delta_4d)
                else:
                    print('Invalid inject type!')
                    exit(12)
    
        if inj_type is not None:
            matmul_out = tf.where(tf.is_nan(matmul_out), tf.ones_like(matmul_out) * 0, matmul_out)
            if 'F32' in inj_type:
                clipped_matmul_out = tf.clip_by_value(matmul_out, -3.402823e38, 3.402823e38)
            elif 'F16' in inj_type:
                clipped_matmul_out = tf.clip_by_value(matmul_out, -65504, 65504)
            elif is_inject and ('I16' in inj_type or 'I8' in inj_type) and ('INPUT' in inj_type or 'WEIGHT' in inj_type or 'PSUM' in inj_type):
                clipped_matmul_out = tf.clip_by_value(matmul_out, quant_min_max[0], quant_min_max[1])
            else:
                clipped_matmul_out = matmul_out
        else:
            clipped_matmul_out = matmul_out

        return clipped_matmul_out


def inj_dense(inputs, units, activation=None, use_bias=True, inj_type=None, inj_layer=None, scope=None, delta_4d=None, quant_min_max=None):
    with tf.name_scope(scope):
        top_name_scope = tf.get_default_graph().get_name_scope()
        is_inject = False
        dense_out = tf.layers.dense(inputs, units, activation=None, use_bias=False)

        if inj_layer and len(inj_layer) > 1:
            print('Error: Do not support injecting to more than 1 layer!')
            exit(12)

        if inj_layer:
            target_layer_compare = inj_layer[0]
            if target_layer_compare and top_name_scope + '/dense/Tensordot/MatMul' == target_layer_compare:
                is_inject = True
                if delta_4d != None and inj_type[:inj_type.find('_')] in ['RD', 'INPUT', 'INPUT_16', 'WEIGHT', 'WEIGHT_16', 'RD_BFLIP', 'RD_PSUM']:
                    dense_out = tf.add(dense_out, delta_4d)
                else:
                    print('Invalid inject_type!')
                    exit(12)
    
        if inj_type is not None:
            dense_out = tf.where(tf.is_nan(dense_out), tf.ones_like(dense_out) * 0, dense_out)
            if 'F32' in inj_type:
                clipped_dense_out = tf.clip_by_value(dense_out, -3.402823e38, 3.402823e38)
            elif 'F16' in inj_type:
                clipped_dense_out = tf.clip_by_value(dense_out, -65504, 65504)
            elif is_inject and ('I16' in inj_type or 'I8' in inj_type) and ('INPUT' in inj_type or 'WEIGHT' in inj_type or 'PSUM' in inj_type):
                clipped_dense_out = tf.clip_by_value(dense_out, quant_min_max[0], quant_min_max[1])
            else:
                clipped_dense_out = dense_out
        else:
            clipped_dense_out = dense_out

        # Add bias or perform activation
        if use_bias:
            with tf.variable_scope(scope):
                layer_bias = tf.get_variable("bias", dtype=tf.float32, shape = [clipped_dense_out.get_shape()[-1]], initializer=tf.zeros_initializer())
            bias_out = tf.nn.bias_add(clipped_dense_out, layer_bias)
        else:
            bias_out = clipped_dense_out
    
        if activation is not None:
            relu_out = activation(bias_out)
        else:
            relu_out = bias_out 
        return relu_out

def get_network_inj_type(precision, inj_type):
    assert precision in ['FP32', 'FP16', 'INT16', 'INT8']
    prec_dict = {
        'FP32': 'F32',
        'FP16': 'F16',
        'INT16': 'I16',
        'INT8': 'I8'
    }
    return inj_type + '_' + prec_dict[precision]

# Converts a binary string to FP16 values
def bin2fp16(bin_str):
    assert len(bin_str) == 16
    sign_bin = bin_str[0]
    if sign_bin == '0':
        sign_val = 1.0
    else:
        sign_val = -1.0
    exponent_bin = bin_str[1:6]
    mantissa_bin = bin_str[6:]
    assert len(mantissa_bin) == 10
    exponent_val = int(exponent_bin,2)
    mantissa_val = 0.0
    for i in range(10):
        if mantissa_bin[i] == '1':
            mantissa_val += pow(2,-i-1)
    # Handling subnormal numbers
    if exponent_val == 0:
        return sign_val * pow(2,-14) * mantissa_val
    # Handling normal numbers
    else:
        value = sign_val * pow(2,exponent_val-15) * (1 + mantissa_val)
        # Handling NaNs and INFs
        if value == 65536:
            return 65535
        elif value == -65536:
            return -65535
        elif value > 65536 or value < -65536:
            return 0
        else:
            return value

# Flip a random bit for one given golden value
def get_perturbation(precision, golden_d, layer, typ=None):
    # Now we only support fp16
    assert 'FP16' == precision

    golden_b = fp162bin(golden_d)
    assert len(golden_b) == 16

    if 'rd' == typ:
        inj_bin = ''
        for _ in range(16):
            inj_bin += str(np.random.randint(0,2))
        perturb = bin2fp16(inj_bin) - golden_d
        flip_bit = 0
    else:
        flip_bit = np.random.randint(16)
        if golden_b[15-flip_bit] == '1':
            inj_b = golden_b[:15-flip_bit] + '0' + golden_b[15-flip_bit+1:]
        else:
            inj_b = golden_b[:15-flip_bit] + '1' + golden_b[15-flip_bit+1:]
        inj_d = bin2fp16(inj_b)
        perturb = inj_d - golden_d

    return flip_bit, perturb

