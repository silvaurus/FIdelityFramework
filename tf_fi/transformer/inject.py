# Written by Yi He
# University of Chicago
# Fault injection for Transformer

import os
import csv
import tensorflow as tf
import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

from data_load import get_batch
from model import Transformer
from model_fp16 import Transformer_fp16
from hparams import Hparams
from utils import get_hypotheses, calc_bleu, calc_return_bleu, postprocess, load_hparams
from inject_util import * 

import logging

logging.basicConfig(level=logging.INFO)

logging.info("# hparams")
hparams = Hparams()
parser = hparams.parser
hp = parser.parse_args()
load_hparams(hp, hp.ckpt)

def str2list(inp, do_float=False):
    str_list = inp.strip("][").split(',')
    return [float(i) if do_float else int(i) for i in str_list]


def one_inject():
    # Obtain target layer information
    layer_dict = {}
    with open(hp.layer_path) as layer_csv:
        layer_reader = csv.reader(layer_csv, delimiter='\t')
        for row in layer_reader:
            layer_dict[row[0]] = row[1]
    print(layer_dict)
    layer_names = [layer_dict["Input tensor name:"], layer_dict["Weight tensor name:"], layer_dict["Output tensor name:"]]
    layer_dims = [str2list(layer_dict["Input shape:"]), str2list(layer_dict["Weight shape:"]), str2list(layer_dict["Output shape:"])]

    logging.info("# Prepare test batches")
    test_batches, num_test_batches, num_test_samples  = get_batch(hp.input_path, hp.input_path,
                                                  100000, 100000,
                                                  hp.vocab, hp.test_batch_size,
                                                  shuffle=False)

    assert num_test_batches == 1

    iter = tf.data.Iterator.from_structure(test_batches.output_types, test_batches.output_shapes)
    xs, ys = iter.get_next()

    test_init_op = iter.make_initializer(test_batches)
     
    delta_inj = tf.placeholder(tf.float16, layer_dims[2])

    logging.info("# Load model")
    if 'FP16' in hp.precision:
        m = Transformer_fp16(hp, inj_type=get_network_inj_type(hp.precision, hp.inj_type), inj_layer=[layer_names[2]])
    else:
        m = Transformer(hp, inj_type=get_network_inj_type(hp.precision, hp.inj_type), inj_layer=[layer_names[2]])

    y_hat, _ = m.eval(xs, ys, delta_inj)

    logging.info("# Session")
    with tf.Session() as sess:
        ckpt_ = tf.train.latest_checkpoint(hp.ckpt)
        ckpt = hp.ckpt if ckpt_ is None else ckpt_ # None: ckpt is a file. otherwise dir.

        all_variables = tf.get_collection_ref(tf.GraphKeys.GLOBAL_VARIABLES)
        v_list = [v for v in all_variables if 'delta' not in v.name]
        saver = tf.train.Saver(var_list=v_list)

        saver.restore(sess, ckpt)
        sess.run(test_init_op)
       
        delta_perturb = np.zeros(shape = layer_dims[2], dtype=np.float32)

        input_tensor = tf.get_default_graph().get_tensor_by_name(layer_names[0] + ":0")
        weight_tensor = tf.get_default_graph().get_tensor_by_name(layer_names[1] + ":0")
        target_tensor = tf.get_default_graph().get_tensor_by_name(layer_names[2] + ":0")
        for _ in range(num_test_batches):
            inp, wt, target = sess.run([input_tensor, weight_tensor, target_tensor], feed_dict={delta_inj: delta_perturb})

        # Run the golden network for testing 
        logging.info("# get hypotheses")
        hypotheses = get_hypotheses(num_test_batches, num_test_samples, sess, y_hat, m.idx2token, delta_inj, delta_perturb)
        logging.info("Golden hypotheses is {}".format(hypotheses))

        ref = [open(hp.output_path, 'r').readlines()[0].split()]
        cc = SmoothingFunction()
        bleu_score = sentence_bleu(ref, hypotheses[0].split(), smoothing_function=cc.method4)
        logging.info('Golden BLEU socre is {}'.format(bleu_score))

        delta_perturb = np.zeros(shape = layer_dims[2], dtype=np.float32)

        if 'INPUT' in hp.inj_type:
            inj_pos_db_set = [[np.random.randint(i)] for i in inp.shape]
        elif 'WEIGHT' in hp.inj_type:
            inj_pos_db_set = [[np.random.randint(i)] for i in wt.shape]
        else:
            inj_pos_db_set = [[np.random.randint(i)] for i in target.shape]
    
        inj_pos = tuple([i[0] for i in inj_pos_db_set])
    
        if 'INPUT' in hp.inj_type:
            golden_d = inp[inj_pos]
            flip_bit, perturb = get_perturbation(hp.precision, golden_d, 'input')
        elif 'WEIGHT' in hp.inj_type:
            golden_d = wt[inj_pos]
            flip_bit, perturb = get_perturbation(hp.precision, golden_d, 'weight')
        elif 'BFLIP' in hp.inj_type:
            golden_d = target[inj_pos]
            flip_bit, perturb = get_perturbation(hp.precision, golden_d, 'target')
        elif 'RD' in hp.inj_type:
            golden_d = target[inj_pos]
            flip_bit, perturb = get_perturbation(hp.precision, golden_d, 'rd')
        else:
            print('Invalid parameter!')
            exit(12)
    
        delta_db_set = [float(flip_bit), perturb]
    
        if 'INPUT' in hp.inj_type:
            inp_perturb = np.zeros(inp.shape)
            inp_perturb[inj_pos] = perturb
            delta_perturb = np.matmul(inp_perturb, wt).reshape(target.shape)
        elif 'WEIGHT' in hp.inj_type:
            wt_perturb = np.zeros(wt.shape)
            wt_perturb[inj_pos] = perturb
            delta_perturb = np.matmul(inp, wt_perturb).reshape(target.shape)
        else:
            delta_perturb = np.zeros(target.shape)
            delta_perturb[inj_pos] = perturb
    
        if '16' in hp.inj_type:
            delta_16 = np.zeros(delta_perturb.shape)
            if 'INPUT' in hp.inj_type:
                random_shape = list(delta_perturb.shape)
                random_shape[-1] = random_shape[-1]//16
                start_index = [np.random.randint(i) for i in random_shape]
                start_index[-1] = start_index[-1]*16
                for i in range(16):
                    start_index = tuple(start_index)
                    delta_16[start_index] = delta_perturb[start_index]
    
                    start_index = list(start_index)
                    start_index[-1] = start_index[-1]+1
            elif 'WEIGHT' in hp.inj_type:
                # Injecting to weights: 16 neurons in 16 batches at a time
                random_shape = list(delta_perturb.shape)
                start_index = tuple([np.random.randint(i) for i in random_shape])
                delta_16[start_index] = delta_perturb[start_index]
            else:
                print('Invalid parameter!')
                exit(12)
    
            delta_perturb = delta_16
    
        # Run the network and get the result
        logging.info("# get hypotheses")
        hypotheses = get_hypotheses(num_test_batches, num_test_samples, sess, y_hat, m.idx2token, delta_inj, delta_perturb)
        logging.info("Inject hypotheses is {}".format(hypotheses))
    
        ref = [open(hp.output_path, 'r').readlines()[0].split()]
        cc = SmoothingFunction()
        bleu_score = sentence_bleu(ref, hypotheses[0].split(), smoothing_function=cc.method4)
        logging.info('Inject BLEU socre is {}'.format(bleu_score))
    


one_inject()
