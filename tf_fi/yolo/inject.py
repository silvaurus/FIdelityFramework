# coding: utf-8

from __future__ import division, print_function

import csv
import tensorflow as tf
import numpy as np
import argparse
import cv2
import re

from utils.misc_utils import parse_anchors, read_class_names
from utils.nms_utils import gpu_nms
from utils.plot_utils import get_color_table, plot_one_box
from utils.data_aug import letterbox_resize

from model_fp16 import yolov3

from inject_util import get_network_inj_type
from inject_util import get_perturbation
from inject_util import perturb_conv

parser = argparse.ArgumentParser(description="YOLO-V3 test single image test procedure.")
parser.add_argument("--image_path", type=str, default=None,
                    help="The path of the input image.")
parser.add_argument("--anchor_path", type=str, default="/home/yiizy/tf_yolo/yolo/data/yolo_anchors.txt",
                    help="The path of the anchor txt file.")
parser.add_argument("--new_size", nargs='*', type=int, default=[416, 416],
                    help="Resize the input image with `new_size`, size format: [width, height]")
parser.add_argument("--letterbox_resize", type=lambda x: (str(x).lower() == 'true'), default=True,
                    help="Whether to use the letterbox resize.")
parser.add_argument("--class_name_path", type=str, default="/home/yiizy/tf_yolo/yolo/data/coco.names",
                    help="The path of the class names.")
parser.add_argument("--restore_path", type=str, default="/home/yiizy/tf_yolo/yolo/data/darknet_weights/yolov3.ckpt",
                    help="The path of the weights to restore.")
parser.add_argument("--plot", type=str, default=False,
                    help="Whether to plot the bounding box")


# Injection parameters
parser.add_argument("--precision", type=str, default='FP16',
                    help="The precision of the network")
parser.add_argument("--inj_type", type=str, default=None,
                    help="The inj type for injection")
parser.add_argument("--layer_path", type=str, default=None,
                    help="The path that stores the layer information")

args = parser.parse_args()

args.anchors = parse_anchors(args.anchor_path)
args.classes = read_class_names(args.class_name_path)
args.num_class = len(args.classes)


def str2list(inp, do_float=False):
    str_list = inp.strip("][").split(',')
    return [float(i) if do_float else int(i) for i in str_list]



layer_dict = {}
with open(args.layer_path) as layer_csv:
    layer_reader = csv.reader(layer_csv, delimiter='\t')
    for row in layer_reader:
        layer_dict[row[0]] = row[1]
layer_names = [layer_dict["Input tensor name:"], layer_dict["Weight tensor name:"], layer_dict["Output tensor name:"]]
layer_dims = [str2list(layer_dict["Input shape:"]), str2list(layer_dict["Weight shape:"]), str2list(layer_dict["Output shape:"])]
layer_stride = int(layer_dict["Layer stride:"])
layer_padding = layer_dict["Layer padding:"]

color_table = get_color_table(args.num_class)

input_image_name = args.image_path
img_ori = cv2.imread(input_image_name)
if args.letterbox_resize:
    img, resize_ratio, dw, dh = letterbox_resize(img_ori, args.new_size[0], args.new_size[1])
else:
    height_ori, width_ori = img_ori.shape[:2]
    img = cv2.resize(img_ori, tuple(args.new_size))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = np.asarray(img, np.float32)
img = img[np.newaxis, :] / 255.

delta_shape = layer_dims[2]
delta_inj = tf.placeholder(tf.float32, delta_shape)

sess = tf.Session()
input_data = tf.placeholder(tf.float32, [1, args.new_size[1], args.new_size[0], 3], name='input_data')
yolo_model = yolov3(args.num_class, args.anchors, inj_type=get_network_inj_type(args.precision, args.inj_type), inj_layer=[layer_names[2]], delta_4d=delta_inj)
with tf.variable_scope('yolov3'):
    pred_feature_maps = yolo_model.forward(input_data, False)
pred_boxes, pred_confs, pred_probs = yolo_model.predict(pred_feature_maps)

pred_scores = pred_confs * pred_probs

boxes, scores, labels = gpu_nms(pred_boxes, pred_scores, args.num_class, max_boxes=200, score_thresh=0.3, nms_thresh=0.45)

all_variables = tf.get_collection_ref(tf.GraphKeys.GLOBAL_VARIABLES)

v_list = [v for v in all_variables if 'weights' not in v.name and 'bias' not in v.name]

saver = tf.train.Saver(var_list=v_list)
saver.restore(sess, args.restore_path)

for variable in all_variables:
    if 'weights' in variable.name or 'bias' in variable.name:
        var = tf.contrib.framework.load_variable(args.restore_path, re.sub('Conv/','',variable.name,1))
        sess.run(variable.assign(var))

# Get golden layer
if layer_names[2] != None:
    input_tensor = tf.get_default_graph().get_tensor_by_name(layer_names[0] + ":0")
    weight_tensor = tf.get_default_graph().get_tensor_by_name(layer_names[1] + ":0")
    target_tensor = tf.get_default_graph().get_tensor_by_name(layer_names[2] + ":0")
            
    delta_perturb = np.zeros(shape=delta_shape, dtype=np.float32)

    inp, wt, target, boxes_, scores_, labels_ = sess.run([input_tensor, weight_tensor, target_tensor, boxes, scores, labels], feed_dict={input_data: img, delta_inj:delta_perturb})

else:
    boxes_, scores_, labels_ = sess.run([boxes, scores, labels], feed_dict={input_data: img})

# rescale the coordinates to the original image
if args.letterbox_resize:
    boxes_[:, [0, 2]] = (boxes_[:, [0, 2]] - dw) / resize_ratio
    boxes_[:, [1, 3]] = (boxes_[:, [1, 3]] - dh) / resize_ratio
else:
    boxes_[:, [0, 2]] *= (width_ori/float(args.new_size[0]))
    boxes_[:, [1, 3]] *= (height_ori/float(args.new_size[1]))

print("********************************************")
print('Here are the golden results: ')
print("box coords:")
print(boxes_)
print('*' * 30)
print("scores:")
print(scores_)
print('*' * 30)
print("labels:")
print(labels_)
print("*******************************************")
   
golden_labels = labels_


if args.plot:
    for i in range(len(boxes_)):
        x0, y0, x1, y1 = boxes_[i]
        plot_one_box(img_ori, [x0, y0, x1, y1], label=args.classes[labels_[i]] + ', {:.2f}%'.format(scores_[i] * 100), color=color_table[labels_[i]])
    cv2.imshow('Detection result', img_ori)
    cv2.imwrite('detection_result.jpg', img_ori)
    cv2.waitKey(0)


def run_one(sess):
    print('Starting one run!')
    delta_perturb = np.zeros(shape = delta_shape, dtype=np.float32)

    if 'INPUT' in args.inj_type:
        t_a, t_b, t_c, t_d = inp.shape
    elif 'WEIGHT' in args.inj_type:
        t_a, t_b, t_c, t_d = wt.shape
    else:
        t_a, t_b, t_c, t_d = target.shape

    p_a = np.random.randint(t_a)
    p_b = np.random.randint(t_b)
    p_c = np.random.randint(t_c)
    p_d = np.random.randint(t_d)
    inj_pos_db_set = [[p_a],[p_b],[p_c],[p_d]]

    if 'INPUT' in args.inj_type:
        golden_d = inp[p_a][p_b][p_c][p_d]
        flip_bit, perturb = get_perturbation(args.precision, golden_d, 'input')
    elif 'WEIGHT' in args.inj_type:
        golden_d = wt[p_a][p_b][p_c][p_d]
        flip_bit, perturb = get_perturbation(args.precision, golden_d, 'weight')
    elif 'BFLIP' in args.inj_type:
        golden_d = target[p_a][p_b][p_c][p_d]
        flip_bit, perturb = get_perturbation(args.precision, golden_d, 'target')
    elif 'RD' in args.inj_type:
        golden_d = target[p_a][p_b][p_c][p_d]
        flip_bit, perturb = get_perturbation(args.precision, golden_d, 'rd')
    else:
        print('Invalid parameter!')
        exit(12)

    delta_db_set = [float(flip_bit), perturb]

    if 'INPUT' in args.inj_type:
        inp_perturb = np.zeros(inp.shape)
        inp_perturb[p_a][p_b][p_c][p_d] = perturb
        delta_perturb = perturb_conv(inp_perturb, wt, layer_stride, layer_padding=='SAME')
    elif 'WEIGHT' in args.inj_type:
        wt_perturb = np.zeros(wt.shape)
        wt_perturb[p_a][p_b][p_c][p_d] = perturb
        delta_perturb = perturb_conv(inp, wt_perturb, layer_stride, layer_padding=='SAME')
    else:
        delta_perturb = np.zeros(target.shape)
        delta_perturb[p_a][p_b][p_c][p_d] = perturb

    if '16' in args.inj_type:
        d_a, d_b, d_c, d_d = delta_perturb.shape
        delta_16 = np.zeros(delta_perturb.shape)

        weight_d = wt.shape[1]
        pad_type = layer_padding
        if pad_type is 'VALID':
            pad = 0
        else:
            pad = weight_d // 2
        stride = layer_stride

        if 'INPUT' in args.inj_type:
            # Injecting to input: 16 neurons in 16 output channels at a time
            start_a = np.random.randint(d_a)
            start_b = np.random.randint(max(0, (p_b+pad)//stride-weight_d+1), min((p_b+pad)//stride+1,d_b))
            start_c = np.random.randint(max(0,(p_c+pad)//stride-weight_d+1), min((p_c+pad)//stride+1,d_c))
            start_d = np.random.randint(d_d // 16)
            for i in range(16):
                delta_16[start_a][start_b][start_c][start_d * 16 + i] = delta_perturb[start_a][start_b][start_c][start_d * 16 + i]
        elif 'WEIGHT' in args.inj_type:
            start_a = np.random.randint(d_a)
            start_p = np.random.randint(d_b * d_c // 16)
            start_d = p_d
            # Injecting to weights: 16 neruons that in continuous 16 hxw plane
            for i in range(16):
                if start_p*16 + i >= d_b * d_c:
                    break
                elem_b = (start_p*16+i) // d_c
                elem_c = (start_p*16+i) % d_c
                delta_16[start_a][elem_b][elem_c][start_d] = delta_perturb[start_a][elem_b][elem_c][start_d]

        delta_perturb = delta_16

    # Run the network with delta_perturb
    boxes_, scores_, labels_ = sess.run([boxes, scores, labels], feed_dict={input_data: img, delta_inj: delta_perturb})

    # rescale the coordinates to the original image
    if args.letterbox_resize:
        boxes_[:, [0, 2]] = (boxes_[:, [0, 2]] - dw) / resize_ratio
        boxes_[:, [1, 3]] = (boxes_[:, [1, 3]] - dh) / resize_ratio
    else:
        boxes_[:, [0, 2]] *= (width_ori/float(args.new_size[0]))
        boxes_[:, [1, 3]] *= (height_ori/float(args.new_size[1]))

    print("*******************************************")
    print('Here are the inject results: ')
    print("box coords:")
    print(boxes_)
    print('*' * 30)
    print("scores:")
    print(scores_)
    print('*' * 30)
    print("labels:")
    print(labels_)
    print("******************************************")


run_one(sess)
