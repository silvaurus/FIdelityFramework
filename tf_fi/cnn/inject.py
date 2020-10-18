import sys
import csv
import argparse
sys.path.append('./slim')
import tensorflow as tf
import numpy as np
import re
from slim.preprocessing.preprocessing_factory import get_preprocessing
from tensorflow.keras.datasets import cifar10

from inj_util import * 

def str2list(inp, do_float=False):
    str_list = inp.strip("][").split(',')
    return [float(i) if do_float else int(i) for i in str_list]

# Inject one error to one layer
def run_one_cnn(args):
    num_inj_per_layer = 1
    # Obtain target layer information
    layer_dict = {}
    with open(args.layer_path) as layer_csv:
        layer_reader = csv.reader(layer_csv, delimiter='\t')
        for row in layer_reader:
            layer_dict[row[0]] = row[1]
    print(layer_dict)
    layer_names = [layer_dict["Input tensor name:"], layer_dict["Weight tensor name:"], layer_dict["Output tensor name:"]]
    layer_dims = [str2list(layer_dict["Input shape:"]), str2list(layer_dict["Weight shape:"]), str2list(layer_dict["Output shape:"])]
    layer_stride = int(layer_dict["Layer stride:"])
    layer_padding = layer_dict["Layer padding:"]
    quant_min_max = str2list(layer_dict["Quant min max:"], True)

    # Obtain delta
    delta_set, inj_pos = delta_generator(args.network, args.precision, args.inj_type, [layer_names[2]], layer_dims[2], quant_min_max)

    # Then start running injection
    tf.reset_default_graph()

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    np_image = x_test[args.image_id]
    image_label = y_test[args.image_id]

    image = tf.placeholder(tf.uint8, shape=[32,32,3])
    if 'ic' in args.network or 'mb' in args.network:
        pre_fn = get_preprocessing('inception', is_training=False)
    else:
        pre_fn = get_preprocessing('resnet_v1_50', is_training=False)

    post_image = pre_fn(image, 224, 224)
    images = tf.expand_dims(post_image, 0)

    # Cast the image for fp16
    if 'fp16' in args.precision:
        images = tf.cast(images, tf.float16)

    # Deploy the network
    arg_scope_fn = get_arg_scope(args.network, args.precision)
    network_fn = get_network(args.network, args.precision)

    # Need to feed in injection relating arguments
    with tf.contrib.slim.arg_scope(arg_scope_fn()):
        net, endpoints = network_fn(images, num_classes=10, is_training=False, inj_type=get_network_inj_type(args.precision, args.inj_type), inj_layer=[layer_names[2]], inj_pos=inj_pos, quant_min_max=quant_min_max)

    # Quantize the network if necessary
    if 'int8' in args.precision:
        tf.contrib.quantize.create_eval_graph()
    elif 'int16' in args.precision:
        tf.contrib.quantize.experimental_create_eval_graph(weight_bits=16, activation_bits=16)

    # Create saver: For FP16, need extra handling for Logits
    all_variables = tf.get_collection_ref(tf.GraphKeys.GLOBAL_VARIABLES)

    if 'fp16' in args.precision:
        v_list = [v for v in all_variables if 'dense' not in v.name and 'delta' not in v.name]
    elif 'rs' in args.network and 'int' in args.precision:
        v_list = [v for v in all_variables if 'delta' not in v.name and 'unit_1/bottleneck_v1/shortcut/act_quant' not in v.name and 'unit_2/bottleneck_v1/conv3/act_quant' not in v.name and 'unit_3/bottleneck_v1/conv3/act_quant' not in v.name and 'unit_4/bottleneck_v1/conv3/act_quant' not in v.name and 'unit_5/bottleneck_v1/conv3/act_quant' not in v.name and 'unit_6/bottleneck_v1/conv3/act_quant' not in v.name]
    else:
        v_list = [v for v in all_variables if 'delta' not in v.name]

    saver = tf.train.Saver(var_list=v_list)

    # Create a session and run it
    with tf.Session() as sess:
        saver.restore(sess, args.ckpt_path)

        # For fp16, restore the dense part of the network 
        if 'fp16' in args.precision:
            dense_var_dict = {'mb': 'MobilenetV1/Logits/Conv2d_1c_1x1/', 'rs': 'resnet_v1_50/logits/', 'ic': 'InceptionV1/Logits/Conv2d_0c_1x1/'}
            for variable in all_variables:
                if 'dense/kernel' in variable.name:
                    var = tf.contrib.framework.load_variable(args.ckpt_path, dense_var_dict[args.network[:2]] + 'weights')
                    sess.run(variable.assign(var[0,0,:,:]))
                if 'dense/bias' in variable.name:
                    var = tf.contrib.framework.load_variable(args.ckpt_path, dense_var_dict[args.network[:2]] + 'biases')
                    sess.run(variable.assign(var))

        elif 'rs' in args.network and 'int' in args.precision:
            for variable in all_variables:
                if 'unit_1/bottleneck_v1/shortcut/act_quant' in variable.name:
                    var = tf.contrib.framework.load_variable(args.ckpt_path, re.sub('act_quant','conv_quant', variable.name))
                    sess.run(variable.assign(var))
                if 'bottleneck_v1/conv3/act_quant' in variable.name and 'unit_1' not in variable.name:
                    var = tf.contrib.framework.load_variable(args.ckpt_path, re.sub('act_quant','conv_quant', variable.name))
                    sess.run(variable.assign(var))

        # If we inject to input/weights or local controls
        if 'INPUT' in args.inj_type or 'WEIGHT' in args.inj_type or 'RD_BFLIP' in args.inj_type:
            layer = layer_names[2]
            delta_np = np.zeros(shape = layer_dims[2], dtype=np.float32)
            scope_string = ''
            if 'mb' in args.network:
                scope_string = 'MobilenetV1' 
            elif 'rs' in args.network:
                scope_string = layer[:layer.rfind('/')]
            else:
                scope_string = layer[layer.find('/')+1:layer.rfind('/')]
            with tf.variable_scope(scope_string ,reuse = True):
                sess.run(tf.get_variable('delta_{}'.format(re.sub('\/','_',layer[layer.rfind('/')+1:])),trainable = False).assign(delta_np))
            # Get this layer's weight
            weight_tensor = tf.get_default_graph().get_tensor_by_name(layer_names[1])
            if 'RD_BFLIP' in args.inj_type:
                # In RD_BFLIP, input tensor means this layer's output
                input_tensor = tf.get_default_graph().get_tensor_by_name(layer + '/Conv2D:0')
            else:
                # Get this layer's input for injecting to input or psum 
                input_tensor = tf.get_default_graph().get_tensor_by_name(layer_names[0])
            # Run the golden network
            wt, inp= sess.run([weight_tensor, input_tensor], feed_dict={image: np_image})
            if 'INPUT' in args.inj_type or 'RD_BFLIP' in args.inj_type:
                if 'INPUT' in args.inj_type or 'RD_BFLIP' in args.inj_type:
                    t_a, t_b, t_c, t_d = inp.shape
                else:
                    t_a, t_b, t_c, t_d = layer_dims[2]
                p_a = np.random.randint(t_a)
                p_b = np.random.randint(t_b)
                p_c = np.random.randint(t_c)
                p_d = np.random.randint(t_d)

                golden_d = inp[p_a][p_b][p_c][p_d]
                if 'RD_BFLIP' in args.inj_type:
                    flip_bit, perturb = get_bit_flip_perturbation(args.network, args.precision, golden_d, layer, 'rd_bflip')
                else:
                    flip_bit, perturb = get_bit_flip_perturbation(args.network, args.precision, golden_d, layer, 'input')

                inp_perturb = np.zeros(inp.shape)
                inp_perturb[p_a][p_b][p_c][p_d] = perturb
                if 'RD_BFLIP' in args.inj_type:
                    delta_perturb = inp_perturb
                else:
                    delta_perturb = perturb_conv(inp_perturb, wt, layer_stride, layer_padding=='SAME', layer_dims[2][-1])
            else:
                t_a, t_b, t_c, t_d = wt.shape
                p_a = np.random.randint(t_a)
                p_b = np.random.randint(t_b)
                p_c = np.random.randint(t_c)
                p_d = np.random.randint(t_d)
                golden_d = wt[p_a][p_b][p_c][p_d]
                flip_bit, perturb = get_bit_flip_perturbation(args.network, args.precision, golden_d, layer, 'weight')
                wt_perturb = np.zeros(wt.shape)
                wt_perturb[p_a][p_b][p_c][p_d] = perturb
                delta_perturb = perturb_conv(inp, wt_perturb, layer_stride, layer_padding=='SAME', layer_dims[2][-1])

            # If we only inject to 16W or 16C we need to reconfig the delta
            if '16' in args.inj_type and 'PSUM' not in args.inj_type:
                _, d_h, d_w, d_c = delta_perturb.shape
                delta_16 = np.zeros(delta_perturb.shape)
                pos_16 = []
                # Injecting to input: 16 neurons in 16 channels at a time
                if 'INPUT' in args.inj_type:
                    weight_d = layer_dims[1]
                    pad_type = layer_padding
                    if pad_type is 'VALID':
                        pad = 0
                    else:
                        pad = weight_d // 2
                    stride = layer_stride
                    start_h = np.random.randint(max(0,(p_b+pad)//stride-weight_d+1), min((p_b+pad)//stride+1,d_h))
                    start_w = np.random.randint(max(0,(p_c+pad)//stride-weight_d+1), min((p_c+pad)//stride+1,d_w))
                    start_c = np.random.randint(d_c // 16)
                    for i in range(16):
                        delta_16[0][start_h][start_w][start_c+i] = delta_perturb[0][start_h][start_w][start_c+i]
                        pos_16.append(start_h)
                        pos_16.append(start_w)
                        pos_16.append(16*start_c+i)
                # Injecting to weight: 16 W at a time
                else:
                    start_p = np.random.randint(d_h * d_w // 16)
                    # It will only affect neurons in p_d
                    start_c = p_d
                    for i in range(16):
                        # If it doesn't have 16, then just break
                        if start_p*16 + i >= d_h * d_w:
                            break
                        elem_h = (start_p*16+i)//d_w
                        elem_w = (start_p*16+i)%d_w
                        delta_16[0][elem_h][elem_w][start_c] = delta_perturb[0][elem_h][elem_w][start_c]
                        pos_16.append(elem_h)
                        pos_16.append(elem_w)
                        pos_16.append(start_c)

                delta_perturb = delta_16
            # Assign delta_perturb back to the variable delta
            with tf.variable_scope(scope_string, reuse=True):
                sess.run(tf.get_variable('delta_{}'.format(re.sub('\/','_',layer[layer.rfind('/')+1:])),trainable = False).assign(delta_perturb))
            # Then run the network again
            if 'mb' in args.network or 'ic' in args.network:
                lgt, prd = sess.run([endpoints['Logits'][0], endpoints['Predictions'][0]], feed_dict={image: np_image})
            else:
                lgt, prd  = sess.run([endpoints['resnet_v1_50/spatial_squeeze'][0], endpoints['predictions'][0]], feed_dict={image: np_image})
       
            # Get a sorted label
            network_labels = np.argsort(lgt)[::-1]

        # If we inject to neuron directly
        else:
            layer = layer_names[2]
            delta_np = np.zeros(shape = layer_dims[2], dtype=np.float32)
            for n_j in range(num_inj_per_layer):
                layer_pos = inj_pos[layer]
                delta_np[0][layer_pos[n_j][0]][layer_pos[n_j][1]][layer_pos[n_j][2]] = delta_set[layer][n_j]
                
            scope_string = ''
            if 'mb' in args.network:
                scope_string = 'MobilenetV1'
            elif 'rs' in args.network:
                scope_string = layer[:layer.rfind('/')]
            else:
                scope_string = layer[layer.find('/')+1:layer.rfind('/')]

            with tf.variable_scope(scope_string ,reuse = True):
                sess.run(tf.get_variable('delta_{}'.format(re.sub('\/','_',layer[layer.rfind('/')+1:])),trainable = False).assign(delta_np))

            op_list = []
            for node in tf.get_default_graph().as_graph_def().node:
                if 'Conv2D' in node.name:
                    op_list.append(node.name + ':0') 

            # Run the network
            if 'mb' in args.network or 'ic' in args.network:
                ops, lgt, prd = sess.run([op_list, endpoints['Logits'][0], endpoints['Predictions'][0]], feed_dict={image: np_image})
            else:
                ops, lgt, prd  = sess.run([op_list, endpoints['resnet_v1_50/spatial_squeeze'][0], endpoints['predictions'][0]], feed_dict={image: np_image})
       
            # Get a sorted label
            network_labels = np.argsort(lgt)[::-1]

    print("After injection, the network label becomes {}".format(network_labels))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', required=True, help="The target network, can be ic, mb or rs. ic --> inception, mb --> Mobilenet, rs --> Resnet.")
    parser.add_argument('--precision', required=True, help="The data precision, can be fp16, int16 or int8")
    parser.add_argument('--inj_type',required=True, help="The injection type, can be INPUT, INPUT16, WEIGHT, WEIGHT16, RD_BFLIP or RD")
    parser.add_argument('--layer_path',required=True, help="Path to file that stores layer information")
    parser.add_argument('--image_id', type=int, required=True, help="The image ID to perform injection")
    parser.add_argument('--ckpt_path', required=True, help="The path to the network checkpoint")

    args = parser.parse_args()

    run_one_cnn(args)


if __name__ == '__main__':
      main()
