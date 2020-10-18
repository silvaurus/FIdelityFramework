import sys
import random
import re
sys.path.append('./slim')
import numpy as np
import struct
import math

from slim.nets import mobilenet_v1
from slim.nets import mobilenet_v1_fp16
from slim.nets import resnet_v1
from slim.nets import resnet_v1_fp16
from slim.nets import inception

# Get a proper network function
def get_network(network, precision):
    network_dict = {
        'mb_fp32': mobilenet_v1.mobilenet_v1, 
        'mb_fp16': mobilenet_v1_fp16.mobilenet_v1, 
        'mb_int16': mobilenet_v1.mobilenet_v1, 
        'mb_int8': mobilenet_v1.mobilenet_v1, 

        'rs_fp32': resnet_v1.resnet_v1_50, 
        'rs_fp16': resnet_v1_fp16.resnet_v1_50, 
        'rs_int16': resnet_v1.resnet_v1_50, 
        'rs_int8': resnet_v1.resnet_v1_50, 

        'ic_fp32': inception.inception_v1, 
        'ic_fp16': inception.inception_v1_fp16, 
        'ic_int16': inception.inception_v1, 
        'ic_int8': inception.inception_v1, 
    }
   
    if 'img' not in network:
        return network_dict[network + '_' + precision]
    else:
        return network_dict[network[:-4] + '_' + precision]
    
# Get proper arg scope for model
def get_arg_scope(network, precision):
    arg_scope_dict = {
        'mb_fp32': mobilenet_v1.mobilenet_v1_arg_scope,
        'mb_fp16': mobilenet_v1_fp16.mobilenet_v1_arg_scope,
        'mb_int16': mobilenet_v1.mobilenet_v1_arg_scope,
        'mb_int8': mobilenet_v1.mobilenet_v1_arg_scope,
        
        'rs_fp32': resnet_v1.resnet_arg_scope,
        'rs_fp16': resnet_v1_fp16.resnet_arg_scope,
        'rs_int16': resnet_v1.resnet_arg_scope,
        'rs_int8': resnet_v1.resnet_arg_scope,
    
        'ic_fp32': inception.inception_v3_arg_scope,
        'ic_fp16': inception.inception_v3_arg_scope,
        'ic_int16': inception.inception_v3_arg_scope,
        'ic_int8': inception.inception_v3_arg_scope,
    }

    if 'img' not in network:
        return arg_scope_dict[network + '_' + precision]
    else:
        return arg_scope_dict[network[:-4] + '_' + precision]


# Converts a binary string to FP32 values
def bin2fp32(bin_str):
    assert len(bin_str) == 32
    data = struct.unpack('!f',struct.pack('!I', int(bin_str, 2)))[0]
    if np.isnan(data):
        return 0
    else:
        return data

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


# Converts a fp32 value to binary
def fp322bin(value):
    return ''.join(bin(c).replace('0b', '').rjust(8, '0') for c in struct.pack('!f', value))
    
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

# Converts a text into 16 bit signed integer
def bin2int16(text):
    assert len(text) == 16
    us_int = int(text,2)
    if us_int > 32767:
        return -(65536 - us_int)
    else:
        return us_int

# Converts a text into 8 bit signed integer
def bin2int8(text):
    assert len(text) == 8
    us_int = int(text,2)
    if us_int > 127:
        return -(256 - us_int)
    else:
        return us_int

# Converts a signed int16 value to binary text
def int162bin(val):
    assert val <= 32767 and val >= -32768
    if val < 0:
        us_val = 65536 + val
    else:
        us_val = val
    return bin(us_val)[2:].zfill(16)

# Converts a signed int8 value to binary text
def int82bin(val):
    assert val <= 127 and val >= -128
    if val < 0:
        us_val = 256 + val
    else:
        us_val = val
    return bin(us_val)[2:].zfill(8)


# Flip a random bit for one given golden value
def get_bit_flip_perturbation(network, precision, golden_d, layer, typ=None, quant_min_max=None):
    if 'fp32' in precision:
        golden_b = fp322bin(golden_d)
        assert len(golden_b) == 32
        flip_bit = np.random.randint(32)
        if golden_b[31-flip_bit] == '1':
            inj_b = golden_b[:31-flip_bit] + '0' + golden_b[31-flip_bit+1:]
        else:
            inj_b = golden_b[:31-flip_bit] + '1' + golden_b[31-flip_bit+1:]
        inj_d = bin2fp32(inj_b)
        perturb = inj_d - golden_d
    elif 'fp16' in precision:
        golden_b =fp162bin(golden_d)
        assert len(golden_b) == 16
        flip_bit = np.random.randint(16)
        if golden_b[15-flip_bit] == '1':
            inj_b = golden_b[:15-flip_bit] + '0' + golden_b[15-flip_bit+1:]
        else:
            inj_b = golden_b[:15-flip_bit] + '1' + golden_b[15-flip_bit+1:]
        inj_d = bin2fp16(inj_b)
        perturb = inj_d - golden_d
    elif 'int16' in precision:
        q_min, q_max = quant_min_max
        granu = (q_max - q_min)/65535
        golden_b = int162bin(max(-32768,min(32767,int(round((golden_d - q_min)/granu)) - 32768)))
        assert len(golden_b) == 16
        flip_bit = np.random.randint(16)
        if golden_b[15-flip_bit] == '1':
            inj_b = golden_b[:15-flip_bit] + '0' + golden_b[15-flip_bit+1:]
        else:
            inj_b = golden_b[:15-flip_bit] + '1' + golden_b[15-flip_bit+1:]
        inj_d = bin2int16(inj_b) + 32768
        perturb = (inj_d * granu + q_min) - golden_d
    elif 'int8' in precision:
        q_min, q_max = quant_min_max
        granu = (q_max - q_min)/256
        golden_b = int82bin(max(-128,min(127,int(round((golden_d - q_min)/granu)) - 128)))
        assert len(golden_b) == 8
        flip_bit = np.random.randint(8)
        if golden_b[7-flip_bit] == '1':
            inj_b = golden_b[:7-flip_bit] + '0' + golden_b[7-flip_bit+1:]
        else:
            inj_b = golden_b[:7-flip_bit] + '1' + golden_b[7-flip_bit+1:]
        inj_d = bin2int8(inj_b) + 128
        perturb = (inj_d * granu + q_min) - golden_d
    else:
        print('Wrong precision!')
        exit(15)
    return flip_bit, perturb
        

# Generates a random delta value given the network, precision and layer
def delta_init(network, precision, layer, quant_min_max):
    if 'fp32' in precision:
        one_bin = ''
        for _ in range(32):
            one_bin += str(np.random.randint(0,2))
        return bin2fp32(one_bin)

    elif 'fp16' in precision:
        one_bin = ''
        for _ in range(16):
            one_bin += str(np.random.randint(0,2))
        return bin2fp16(one_bin)

    elif 'int8' in precision:
        quant_min, quant_max = quant_min_max
        delta_int = np.random.randint(0,pow(2,8))
        return delta_int * ((quant_max - quant_min) / 255) + quant_min
    

    elif 'int16' in precision:
        quant_min, quant_max = quant_min_max
        delta_int = np.random.randint(0,pow(2,16))
        return delta_int * ((quant_max - quant_min) / 65535) + quant_min


# Generate delta for one injection
def delta_generator(network, precision, inj_type, layer_list, layer_dim, quant_min_max=None):
    num_inj_per_layer = 1
    delta_set = {}
    inj_pos = {}
    inj_h_set = []
    inj_w_set = []
    inj_c_set = []
    
    if 'INPUT' not in inj_type and 'WEIGHT' not in inj_type and 'RD_BFLIP' not in inj_type:
        for layer in layer_list: 
            tup_set = []
            layer_delta_set = []
            _, max_h, max_w, max_c = layer_dim
            while len(tup_set) < num_inj_per_layer:
                tup = (random.randint(0,max_h-1), random.randint(0,max_w-1), random.randint(0,max_c-1))
                if tup not in tup_set:
                    tup_set.append(tup)
                    inj_h_set.append(tup[0])
                    inj_w_set.append(tup[1])
                    inj_c_set.append(tup[2])

                    # Initialize delta
                    delta_val = delta_init(network, precision, layer, quant_min_max)
                    layer_delta_set.append(delta_val)

            inj_pos[layer] = tup_set
            delta_set[layer] = layer_delta_set

    return delta_set, inj_pos

# Convolution for a input and a weight
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


# Obtain one partial sum in output position inj_pos
def get_psum(inp, weight, bound_type, stride, same_padding, inj_pos):
    inj_h = inj_pos[0][0]
    inj_w = inj_pos[1][0]
    inj_k = inj_pos[2][0]

    total_c = inp.shape[3]
    c_start = 0
    c_end = 0
    assert 'RD_PSUM' in bound_type
    if '_4' in bound_type:
        c_start = np.random.randint((total_c // 4)) * 4
        c_end = c_start + 4
    elif '_16' in bound_type:
        c_start = np.random.randint((total_c // 16)) * 16
        c_end = c_start + 16
    elif '_64' in bound_type:
        c_start = np.random.randint((total_c // 64)) * 64
        c_end = c_start + 64
    else:
        print('Error bound_type!')
        exit()
    
    pad_inp = inp
    if same_padding is True:
        pad_size = int(weight.shape[1]/2)
        if pad_size > 0:
            pad_inp = np.pad(inp, pad_width=((0,0),(pad_size,pad_size),(pad_size,pad_size),(0,0)), mode='constant', constant_values=0)
    assert (pad_inp.shape[1] - weight.shape[0] + 1) % stride == 0

    # Based on the output neuron position get the right pad_inp area
    inp_h_start = inj_h * stride
    inp_h_end = inp_h_start + weight.shape[0]
    inp_w_start = inj_w * stride
    inp_w_end = inp_w_start + weight.shape[0]
    inp_block = pad_inp[0, inp_h_start:inp_h_end, inp_w_start:inp_w_end, c_start:c_end]
    weight_block = weight[:, :, c_start:c_end, inj_k]

    # Do convolution
    psum_result = np.sum(np.multiply(inp_block, weight_block))

    return psum_result

# Correlates the interface's bound type with network's inj_type
def get_network_inj_type(precision, inj_type):
    assert precision in ['fp32', 'fp16', 'int16', 'int8']
    prec_dict = {
        'fp32': 'F32',
        'fp16': 'F16',
        'int16': 'I16',
        'int8': 'I8'
    }
    return inj_type + prec_dict[precision]

