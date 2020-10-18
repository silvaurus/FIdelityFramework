#!/bin/python3
# Written by Yi He
# University of Chicago
# Utilities to support network neuron injections

import tensorflow as tf
import numpy as np
import math

# Creates a fp16 variable scope, such that each variable designed under this scope will be casted to fp16
def custom_dtype_getter(getter, name, shape=None, dtype=tf.float16,
                           *args, **kwargs):
    #Creates variables in fp32, then casts to fp16 if necessary.
    var = getter(name, shape, tf.float32, *args, **kwargs)
    return tf.cast(var, dtype=tf.float16, name=name + '_cast')


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
def fp16_dtype_getter(getter, name, shape=None, dtype=tf.float16,
                           *args, **kwargs):
    #Creates variables in fp32, then casts to fp16 if necessary.
    var = getter(name, shape, tf.float32, *args, **kwargs)
    return tf.cast(var, dtype=tf.float16, name=name + '_cast')

def fp32_dtype_getter(getter, name, shape=None, dtype=tf.float32,
                           *args, **kwargs):
    #Creates variables in fp32, then casts to fp16 if necessary.
    return getter(name, shape, tf.float32, *args, **kwargs)


def get_network_inj_type(precision, inj_type):
    if inj_type is None:
        return None
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
def get_perturbation(precision, golden_d, typ=None):
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

# perform perturb convolution
def perturb_conv(inp, weight, stride, same_padding, wt_exp=None):
    pad_inp = inp
    if same_padding is True:
        pad_size = int(weight.shape[1]/2)
        if pad_size > 0:
            pad_inp = np.pad(inp, pad_width=((0,0),(pad_size,pad_size),(pad_size,pad_size),(0,0)), mode='constant', constant_values=0)
    assert (pad_inp.shape[1] - weight.shape[0] + 1) % stride == 0

    # Expand the weights for depthwise convolution
    if weight.shape[3] == 1 and wt_exp is not None:
        exp_weight = np.tile(weight, wt_exp)
    else:
        exp_weight = weight
    
    wt_w = exp_weight.shape[0]
    wt_c = exp_weight.shape[2]
    
    out_b = pad_inp.shape[0]
    out_w = int((pad_inp.shape[1] - exp_weight.shape[0] + 1) / stride)
    out_c = exp_weight.shape[3]
    
    out_tensor = np.zeros((out_b, out_w, out_w, out_c))
    for o_b in range(out_b):
        for o_h in range(out_w):
            for o_w in range(out_w):
                for o_c in range(out_c):
                    target_neuron = np.sum(np.multiply(pad_inp[o_b,o_h*stride:o_h*stride+wt_w,o_w*stride:o_w*stride+wt_w,:],exp_weight[:,:,:,o_c]))
                    out_tensor[o_b][o_h][o_w][o_c] = target_neuron
    return out_tensor

