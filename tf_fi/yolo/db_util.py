#!/bin/python3
# Written by Yi He
# University of Chicago
# This script provides useful functions for postgresql DB operations

import sys
sys.path.append('/home/yiizy/tf_nn_inject/slim')
import tensorflow as tf
import math
import os
import random
import numpy as np
import re
import psycopg2


#######################################
# The function that transforms a list to database string array
# Args: input_list: The input list
# Return: A text string that can be sent to database function

def list_to_db_array(input_list):
    if len(input_list) is 0:
        return 'ARRAY[]::double precision[]'
    out_txt = 'ARRAY['
    for n in range(len(input_list) - 1):
        if math.isnan(input_list[n]):
            out_txt += '\'nan\'::double precision,'
        elif math.isinf(input_list[n]):
            if input_list[n] > 0:
                out_txt += '\'infinity\'::double precision,'
            else:
                out_txt += '\'-infinity\'::double precision,'
        else:
            out_txt += str(input_list[n]) + ','

    if math.isnan(input_list[-1]):
        out_txt += '\'nan\'::double precision]'
    elif math.isinf(input_list[-1]):
        if input_list[-1] > 0:
            out_txt += '\'infinity\'::double precision]'
        else:
            out_txt += '\'-infinity\'::double precision]'
    else:
        out_txt += str(input_list[-1]) + ']'
    return out_txt


############################################
# This function pushes one injection run results to database
# Args: inj_id: The injection id in database (primary key)
#       network: The target network
#       precision: The precision used
#       bound_type: The bound type used for injection
#       network_labels: The injection output's ranked labels
#       lgt: Logits of the network output
#       prd: Probability of the network output
#       delta_db_set: The set of delta values
#       inj_pos_db_set: The injection positions
# Return: Nothing (Write to database)

def push_inject_one(inj_id, network, precision, bound_type, boxes, scores, labels, delta_db_set, inj_pos_db_set, golden_labels):
    # Try to connect
    try:
        conn=psycopg2.connect(database='neuron_inject', user='yiizy', password='21stCenszMan!', host='yanjing-compute1.cs.uchicago.edu')
    except:
        print('Unable to connect to the database.')

    cur = conn.cursor()

    if 'val' in network:
        exe_str = """UPDATE {}_{}_{}_inject SET job_status = 'DONE', box_coords = {}, scores = {}, labels = {}, inj_h_set = {}, inj_w_set = {}, inj_c_set = {}, delta_set = {} WHERE inj_id = {}""".format(network, precision, bound_type, list_to_db_array(boxes), list_to_db_array(scores), list_to_db_array(labels), list_to_db_array(inj_pos_db_set[0]), list_to_db_array(inj_pos_db_set[1]), list_to_db_array(inj_pos_db_set[2]), list_to_db_array(delta_db_set), inj_id)
    else:
        exe_str = """UPDATE {}_{}_{}_inject SET job_status = 'DONE', box_coords = {}, scores = {}, labels = {}, golden_labels = {}, inj_h_set = {}, inj_w_set = {}, inj_c_set = {}, delta_set = {} WHERE inj_id = {}""".format(network, precision, bound_type, list_to_db_array(boxes), list_to_db_array(scores), list_to_db_array(labels), list_to_db_array(golden_labels), list_to_db_array(inj_pos_db_set[0]), list_to_db_array(inj_pos_db_set[1]), list_to_db_array(inj_pos_db_set[2]), list_to_db_array(delta_db_set), inj_id)
    print(exe_str)
    cur.execute(exe_str)
    conn.commit()


def push_mixed_mode_one(inj_id, network, precision, bound_type, boxes, scores, labels):
    try:
        conn=psycopg2.connect(database='neuron_inject', user='yiizy', password='21stCenszMan!', host='yanjing-compute1.cs.uchicago.edu')
    except:
        print('Unable to connect to the database.')

    cur = conn.cursor()
    exe_str = """INSERT INTO yolo_relu_fp16_mix_mode (inj_id, box_coords, scores, labels) VALUES ({}, {}, {}, {})""".format(inj_id, list_to_db_array(boxes), list_to_db_array(scores), list_to_db_array(labels))
    print(exe_str)
    cur.execute(exe_str)
    conn.commit()

###########################################
# This function pushes the shape of delta
# #########################################
def push_delta_shape(line_id, layer, target_shape, input_shape, weight_shape):
    # Try to connect
    try:
        conn=psycopg2.connect(database='neuron_inject', user='yiizy', password='21stCenszMan!', host='yanjing-compute1.cs.uchicago.edu')
    except:
        print('Unable to connect to the database.')

    cur = conn.cursor()

    exe_str = """INSERT INTO transformer_delta_size (line_id, inj_layer, delta_shape, input_shape, weights_shape) VALUES ({},'{}', {}, {}, {});""".format(line_id, layer, list_to_db_array(target_shape), list_to_db_array(input_shape), list_to_db_array(weight_shape))
    print(exe_str)
    cur.execute(exe_str)
    conn.commit()



########################################
# This function fetches one job from database, set its state to running
# Args: network: The network to be injected
#       precision: The data precision used
#       bound_type: The bound type used in injection (normally we have RD for random injection, RB for inference-bounded injection)
# Return: inj_id: The injection id (primary key of the database)
#         image_id: The image for injection
#         layer_list: The list of inject layers
#         num_inj: Number of injections done on each inject layer

def fetch_one_job_from_db(network, precision, bound_type, inj_layer=None, image_id=None):
    # Try to connect
    try:
        conn=psycopg2.connect(database='neuron_inject', user='yiizy', password='21stCenszMan!', host='yanjing-compute1.cs.uchicago.edu')
    except:
        print('Unable to connect to the database.')

    # Get one job
    cur = conn.cursor()
    if not inj_layer and not image_id:
        exe_str = """UPDATE {}_{}_{}_inject SET job_status = 'RUNNING' WHERE inj_id = (SELECT inj_id FROM {}_{}_{}_inject WHERE job_status = 'READY' LIMIT 1 FOR UPDATE SKIP LOCKED) RETURNING inj_id, image_id, layer_list, num_inj;""".format(network, precision, bound_type, network, precision, bound_type)
    else:
        if image_id == 'val':
            img_id = -1
            exe_str = """UPDATE {}_{}_{}_inject SET job_status = 'RUNNING' WHERE inj_id = (SELECT inj_id FROM {}_{}_{}_inject WHERE layer_list = ARRAY['{}'] AND image_id = {} AND job_status = 'READY' LIMIT 1 FOR UPDATE SKIP LOCKED) RETURNING inj_id, image_id, layer_list, num_inj;""".format(network, precision, bound_type, network, precision, bound_type, inj_layer, img_id)
        else:
            exe_str = """UPDATE {}_{}_{}_inject SET job_status = 'RUNNING', layer_list = ARRAY['{}'] WHERE inj_id = (SELECT inj_id FROM {}_{}_{}_inject WHERE image_id = {} AND job_status = 'READY' LIMIT 1 FOR UPDATE SKIP LOCKED) RETURNING inj_id, image_id, layer_list, num_inj;""".format(network, precision, bound_type, inj_layer, network, precision, bound_type, int(image_id))

    print(exe_str)
    cur.execute(exe_str)
    job = cur.fetchall()[0]
    conn.commit()

    inj_id = job[0]
    image_id = job[1]
    layer_list = job[2]
    num_inj = job[3]
    
    if not inj_layer:
        return inj_id, image_id , layer_list, int(num_inj)
    else:
        return inj_id, int(num_inj)


# Doesn't support integer for now
def get_quant_min_max(network, precision, layer):
    return None


# The function to get a bound for each layer
def get_db_bound(network, precision, bound_type, layer_list):
    if 'int' in precision and ('INPUT' in bound_type or 'WEIGHT' in bound_type or 'PSUM' in bound_type):
        bound = list(get_quant_min_max(network, precision, layer_list[0]))
        print('Bound is {}'.format(bound))
        return bound
    else:
        return None


def db_get_mixed_mode_inj_id(network, precision, bound_type):
    try:
        conn=psycopg2.connect(database='neuron_inject', user='yiizy', password='21stCenszMan!', host='yanjing-compute1.cs.uchicago.edu')
    except:
        print('Unable to connect to the database.')

    cur = conn.cursor()
    cur.execute("""SELECT inj_id FROM hw_v1_fp16_yolo_injections WHERE job_status = ANY(ARRAY['ALL_GOOD','NO_DIFF_FILE','WRONG_DIFF_FORMAT']) AND prog_status = 'PASS' AND num_output_diff > 0;""")
    result = cur.fetchall()
    all_id = [x[0] for x in result]

    cur.execute("""SELECT inj_id FROM yolo_relu_fp16_mix_mode;""")
    result = cur.fetchall()
    done_id = [x[0] for x in result]

    return [x for x in all_id if x not in done_id]


# Function that returns the diff_position, golden value and inj_value for a given inj_id
def db_get_diff_output(inj_id):
    try:
        conn=psycopg2.connect(database='neuron_inject', user='yiizy', password='21stCenszMan!', host='yanjing-compute1.cs.uchicago.edu')
    except:
        print('Unable to connect to the database.')

    cur = conn.cursor()
    cur.execute("""SELECT diff_output_indexes, golden_diff_output, inject_diff_output FROM hw_v1_fp16_yolo_injections WHERE inj_id = {}""".format(inj_id))
    result = cur.fetchall()
    return result[0][0], result[0][1], result[0][2]


