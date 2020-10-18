#!/bin/python3

import tensorflow as tf
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

ckpt_file = '/home/yiizy/tf_yolo/yolo/data/darknet_weights/yolov3.ckpt'

print_tensors_in_checkpoint_file(ckpt_file, all_tensors=True, tensor_name='')
